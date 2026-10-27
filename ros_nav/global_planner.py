"""ROS-free global planner: A* on the occupancy grid (+ remembered lidar circles).

    gp = GlobalPlanner(gridmap)
    gp.set_goal((x, y))
    ok = gp.update(pose_xy, dynamic_circles)       # (re)plans only when needed
    pt, yaw, remaining, is_final = gp.lookahead(pose_xy, horizon)

The controller uses lookahead() to pick the local goal that TEB optimises towards, so TEB
never has to find its way around a wall by itself. The same global path is the natural
extra observation for the RL policy later.

Costs (distance from the nearest obstacle):  < r_hard  blocked,  r_hard..r_near  x6,
r_near..r_far  x2,  beyond  x1.  r_hard = robot radius + small margin, so a goal the node
accepts (>= ROBOT_RADIUS + 0.05 from a wall) is always reachable.
Dynamic obstacles are circles [x, y, r] (the ObstacleTracker output); they are treated
exactly like map cells.
"""
import heapq
import math

import numpy as np

_SQ2 = math.sqrt(2.0)
_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, _SQ2), (-1, 1, _SQ2), (1, -1, _SQ2), (1, 1, _SQ2)]


def _disk_dilate(mask, rad_cells):
    """OR of `mask` shifted by every offset inside a disk (no wrap-around)."""
    out = mask.copy()
    H, W = mask.shape
    k = int(math.ceil(rad_cells))
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            if (dr == 0 and dc == 0) or dr * dr + dc * dc > rad_cells * rad_cells:
                continue
            out[max(0, dr):min(H, H + dr), max(0, dc):min(W, W + dc)] |= \
                mask[max(0, -dr):min(H, H - dr), max(0, -dc):min(W, W - dc)]
    return out


class GlobalPlanner:
    def __init__(self, gridmap, r_hard=0.30, r_near=0.50, r_far=0.70,
                 dyn_near=0.40, dyn_far=0.50,
                 path_step=0.10, max_goal_snap=1.0, deviation=1.0):
        """r_hard/r_near/r_far : clearances around map obstacles (blocked / x6 cost / x2 cost).
        dyn_near/dyn_far      : same bands around sensed circles. Tighter than the map ones on
                                purpose: the circles already include the tracker's own padding,
                                and r_hard is shared so the blocked zone is identical."""
        self.dyn_near, self.dyn_far = dyn_near, dyn_far
        self.map = gridmap
        self.res = gridmap.resolution
        self.ox, self.oy = gridmap.origin[0], gridmap.origin[1]
        self.H, self.W = gridmap.occ.shape
        self.r_hard, self.r_near, self.r_far = r_hard, r_near, r_far
        self.path_step = path_step
        self.max_goal_snap = max_goal_snap
        self.deviation = deviation

        occ = gridmap.occ
        self._hard_s = _disk_dilate(occ, r_hard / self.res)
        self._near_s = _disk_dilate(occ, r_near / self.res)
        self._far_s = _disk_dilate(occ, r_far / self.res)
        self.hard = self._hard_s
        self.cost = self._make_cost(self._hard_s, self._near_s, self._far_s)

        self._goal = None
        self.effective_goal = None  # == goal, or the nearest free point if the goal got blocked
        self._circles_key = None
        self.path = None        # (k, 2) densified, ends at the true goal
        self.s = None           # cumulative arclength of path
        self._idx = 0           # progress along the path (monotonic)
        self.version = 0        # bumped every time a new path is computed
        self.status = "idle"

    # ------------------------------------------------------------------ grid helpers
    @staticmethod
    def _make_cost(hard, near, far):
        cost = np.ones(hard.shape, np.float32)
        cost[far] = 2.0
        cost[near] = 6.0
        cost[hard] = np.inf
        return cost

    def to_cell(self, xy):
        c = int(math.floor((xy[0] - self.ox) / self.res))
        r = int(math.floor((xy[1] - self.oy) / self.res))
        return r, c

    def to_world(self, r, c):
        return self.ox + (c + 0.5) * self.res, self.oy + (r + 0.5) * self.res

    def _inb(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W

    def _blocked_xy(self, xy, mask):
        r, c = self.to_cell(xy)
        return (not self._inb(r, c)) or bool(mask[r, c])

    def _line_clear(self, a, b, mask):
        """True if the straight segment a->b (world xy) stays out of `mask`."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(math.ceil(np.linalg.norm(b - a) / (0.5 * self.res))) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts = a[None, :] + t[:, None] * (b - a)[None, :]
        c = np.floor((pts[:, 0] - self.ox) / self.res).astype(int)
        r = np.floor((pts[:, 1] - self.oy) / self.res).astype(int)
        if np.any((c < 0) | (c >= self.W) | (r < 0) | (r >= self.H)):
            return False
        return not bool(mask[r, c].any())

    # ------------------------------------------------------------------ dynamic layer
    def _apply_circles(self, circles):
        """Rebuild hard/cost = static layers + circles [x, y, r]."""
        hard = self._hard_s.copy()
        near, far = self._near_s.copy(), self._far_s.copy()
        for x, y, rad in circles:
            reach = rad + max(self.r_hard, self.dyn_far)
            c0 = max(0, int(math.floor((x - reach - self.ox) / self.res)))
            c1 = min(self.W, int(math.ceil((x + reach - self.ox) / self.res)) + 1)
            r0 = max(0, int(math.floor((y - reach - self.oy) / self.res)))
            r1 = min(self.H, int(math.ceil((y + reach - self.oy) / self.res)) + 1)
            if c0 >= c1 or r0 >= r1:
                continue
            cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
            d = np.hypot(self.ox + (cc + 0.5) * self.res - x,
                         self.oy + (rr + 0.5) * self.res - y) - rad
            hard[r0:r1, c0:c1] |= d < self.r_hard
            near[r0:r1, c0:c1] |= d < self.dyn_near
            far[r0:r1, c0:c1] |= d < self.dyn_far
        self.hard = hard
        self.cost = self._make_cost(hard, near, far)

    # ------------------------------------------------------------------ public API
    def set_goal(self, goal_xy):
        self._goal = np.asarray(goal_xy[:2], float)
        self.path = self.s = None
        self._idx = 0
        self.status = "new goal"

    def clear(self):
        self._goal = None
        self.path = self.s = None
        self.status = "idle"

    def update(self, pose_xy, circles=None):
        """Make sure a valid path exists. Returns True if one is available."""
        if self._goal is None:
            return False
        pose_xy = np.asarray(pose_xy[:2], float)
        circles = np.empty((0, 3)) if circles is None else np.asarray(circles, float).reshape(-1, 3)
        key = tuple(np.round(circles, 2).ravel())
        changed = key != self._circles_key
        if changed:
            self._circles_key = key
            self._apply_circles(circles)

        need = self.path is None
        reason = "new goal" if need else ""
        if not need and changed and self._path_blocked():
            need, reason = True, "path blocked by new obstacle"
        if not need and self._dist_to_path(pose_xy) > self.deviation:
            need, reason = True, "off path"
        if need:
            ok = self._replan(pose_xy)
            self.status = f"planned ({reason})" if ok else "no path"
            return ok
        return True

    def lookahead(self, pose_xy, horizon):
        """Local goal for the local planner.

        Returns (xy, tangent_yaw, remaining_arclength, is_final).
        xy is the furthest point of the path within `horizon` that the robot can see in a
        straight line (so the local planner's initial guess never cuts through a wall).
        """
        pose_xy = np.asarray(pose_xy[:2], float)
        self._advance(pose_xy)
        s0 = self.s[self._idx]
        remaining = float(self.s[-1] - s0)
        goal = self.path[-1]
        if remaining <= horizon and self._line_clear(pose_xy, goal, self.hard):
            return goal.copy(), self._tangent(len(self.path) - 1), remaining, True

        last = int(np.searchsorted(self.s, s0 + horizon, side="right")) - 1
        last = min(max(last, self._idx + 1), len(self.path) - 1)
        for k in range(last, self._idx, -1):
            if self._line_clear(pose_xy, self.path[k], self.hard):
                return self.path[k].copy(), self._tangent(k), remaining, k == len(self.path) - 1
        k = min(self._idx + 3, len(self.path) - 1)   # robot squeezed: follow the path itself
        return self.path[k].copy(), self._tangent(k), remaining, k == len(self.path) - 1

    # ------------------------------------------------------------------ path bookkeeping
    def _tangent(self, k):
        j = min(k + 3, len(self.path) - 1)
        i = max(0, j - 3) if j == len(self.path) - 1 else k
        d = self.path[j] - self.path[i]
        if np.linalg.norm(d) < 1e-6:
            d = self.path[-1] - self.path[max(0, len(self.path) - 4)]
        return float(math.atan2(d[1], d[0]))

    def _advance(self, xy, window=60):
        hi = min(len(self.path), self._idx + window)
        d = np.linalg.norm(self.path[self._idx:hi] - xy, axis=1)
        self._idx += int(np.argmin(d))

    def _dist_to_path(self, xy):
        self._advance(xy)
        return float(np.linalg.norm(self.path[self._idx] - xy))

    def _path_blocked(self, ahead=6.0):
        hi = int(np.searchsorted(self.s, self.s[self._idx] + ahead, side="right"))
        pts = self.path[self._idx:hi]
        c = np.floor((pts[:, 0] - self.ox) / self.res).astype(int)
        r = np.floor((pts[:, 1] - self.oy) / self.res).astype(int)
        ok = (c >= 0) & (c < self.W) & (r >= 0) & (r < self.H)
        return bool(self.hard[r[ok], c[ok]].any())

    # ------------------------------------------------------------------ planning
    def _nearest_free(self, rc, max_dist):
        """Nearest non-hard cell to rc (BFS ring search), or None."""
        r0, c0 = rc
        if self._inb(r0, c0) and not self.hard[r0, c0]:
            return rc
        k = int(math.ceil(max_dist / self.res))
        best, best_d = None, 1e9
        for r in range(max(0, r0 - k), min(self.H, r0 + k + 1)):
            for c in range(max(0, c0 - k), min(self.W, c0 + k + 1)):
                if not self.hard[r, c]:
                    d = (r - r0) ** 2 + (c - c0) ** 2
                    if d < best_d:
                        best, best_d = (r, c), d
        return best if best_d <= k * k else None

    def _replan(self, pose_xy):
        self.path = None
        gcell = self.to_cell(self._goal)
        if not self._inb(*gcell):
            return False
        start = self._nearest_free(self.to_cell(pose_xy), 1.0)
        goal = self._nearest_free(gcell, self.max_goal_snap)
        if start is None or goal is None:
            return False
        cells = self._astar(start, goal)
        if cells is None:
            return False
        pts = np.array([self.to_world(r, c) for r, c in cells])
        pts = self._string_pull(pts)
        # goal blocked (e.g. a newly sensed obstacle sits on it): end at the nearest free spot
        self.effective_goal = self._goal.copy() if goal == gcell else np.array(self.to_world(*goal))
        poly = np.vstack([pose_xy, pts[1:] if len(pts) > 1 else pts, self.effective_goal])
        self.path = self._densify(poly)
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self._idx = 0
        self.version += 1
        return True

    def _astar(self, start, goal):
        H, W = self.H, self.W
        hard = self.hard.ravel().tolist()
        cost = self.cost.ravel().tolist()
        s, g = start[0] * W + start[1], goal[0] * W + goal[1]
        gr, gc = goal
        INF = float("inf")
        gs = [INF] * (H * W)
        parent = [-1] * (H * W)
        closed = bytearray(H * W)
        gs[s] = 0.0
        heap = [(0.0, s)]
        while heap:
            _, u = heapq.heappop(heap)
            if closed[u]:
                continue
            if u == g:
                out = []
                while u != -1:
                    out.append((u // W, u % W))
                    u = parent[u]
                return out[::-1]
            closed[u] = 1
            ur, uc = divmod(u, W)
            for dr, dc, step in _NEIGHBOURS:
                vr, vc = ur + dr, uc + dc
                if not (0 <= vr < H and 0 <= vc < W):
                    continue
                v = vr * W + vc
                if closed[v] or hard[v]:
                    continue
                if dr and dc and (hard[ur * W + vc] or hard[vr * W + uc]):
                    continue  # no corner cutting
                ng = gs[u] + step * 0.5 * (cost[u] + cost[v])
                if ng < gs[v]:
                    gs[v] = ng
                    parent[v] = u
                    dx, dy = abs(vc - gc), abs(vr - gr)
                    h = (dx + dy) + (_SQ2 - 2.0) * min(dx, dy)
                    heapq.heappush(heap, (ng + h, v))
        return None

    def _string_pull(self, pts):
        """Greedy line-of-sight shortcuts that keep >= r_near clearance."""
        if len(pts) < 3:
            return pts
        wide = self.cost > 2.0 if self.cost is not None else self.hard   # hard + near band
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = i + 1
            while j + 1 < len(pts) and self._line_clear(pts[i], pts[j + 1], wide):
                j += 1
            out.append(pts[j])
            i = j
        return np.array(out)

    def _densify(self, poly):
        out = [poly[0]]
        for a, b in zip(poly[:-1], poly[1:]):
            d = float(np.linalg.norm(b - a))
            n = max(1, int(math.ceil(d / self.path_step)))
            for k in range(1, n + 1):
                out.append(a + (b - a) * (k / n))
        return np.array(out)

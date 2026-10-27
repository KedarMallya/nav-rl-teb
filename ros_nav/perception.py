"""Random hidden obstacles + simulated 2-D lidar + a tiny obstacle tracker.

The static map is known to the planner up front. The random obstacles exist only in MuJoCo.
The lidar (mj_ray) sees everything; every hit that is NOT explained by the known map becomes
an "unknown obstacle", is clustered into a circle, and is remembered (static-world assumption)
so the planner keeps avoiding it after it leaves the sensor's view.
"""
import threading

import mujoco
import numpy as np

from map_utils import distance_to_rects


# ---------------------------------------------------------------------------
# random obstacles (ground truth, put into MuJoCo only)
# ---------------------------------------------------------------------------
def random_obstacles(rects, bounds, n, seed, keep_out=(), keep_out_clear=1.2,
                     wall_clear=0.8, gap=0.8):
    """Rows [kind, x, y, sx, sy, yaw]; kind 0 = cylinder (sx = radius), 1 = box (half sizes sx, sy).

    Placed in free space: >= wall_clear from map walls, >= gap from each other and
    >= keep_out_clear from every (x, y, r) in keep_out (e.g. the start pose).
    """
    rng = np.random.default_rng(seed)
    xmin, ymin, xmax, ymax = bounds
    out, placed = [], []
    for _ in range(max(1, n) * 300):
        if len(out) >= n:
            break
        x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        if rng.random() < 0.5:
            kind, sx, sy, yaw = 0, rng.uniform(0.15, 0.30), 0.0, 0.0
            r = sx
        else:
            kind, sx, sy, yaw = 1, rng.uniform(0.15, 0.35), rng.uniform(0.15, 0.35), rng.uniform(0, np.pi)
            r = float(np.hypot(sx, sy))
        if distance_to_rects((x, y), rects) < r + wall_clear:
            continue
        if any(np.hypot(x - kx, y - ky) < r + kr + keep_out_clear for kx, ky, kr in keep_out):
            continue
        if any(np.hypot(x - px, y - py) < r + pr + gap for px, py, pr in placed):
            continue
        out.append([kind, x, y, sx, sy, yaw])
        placed.append((x, y, r))
    return np.array(out, float).reshape(-1, 6)


def extras_to_circles(extras):
    """(k,6) extras -> (k,3) [x, y, r] bounding circles."""
    if len(extras) == 0:
        return np.empty((0, 3))
    r = np.where(extras[:, 0] == 0, extras[:, 3], np.hypot(extras[:, 3], extras[:, 4]))
    return np.column_stack([extras[:, 1], extras[:, 2], r])


# ---------------------------------------------------------------------------
# lidar
# ---------------------------------------------------------------------------
class Lidar:
    def __init__(self, model, data, base_body="base", n_rays=360, max_range=4.0, height=0.25):
        self.model, self.data = model, data
        self.body = model.body(base_body).id
        self.n, self.max_range, self.height = n_rays, max_range, height
        self.angles = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)  # relative to heading
        self._gid = np.zeros(1, dtype=np.int32)

    def scan(self, pose):
        """Ranges [m] per ray (inf = no hit within range). Call from the thread that steps MuJoCo."""
        x, y, yaw = pose
        pnt = np.array([x, y, self.height])
        a = self.angles + yaw
        dirs = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)
        ranges = np.full(self.n, np.inf)
        for i in range(self.n):
            d = mujoco.mj_ray(self.model, self.data, pnt, dirs[i], None, 1, self.body, self._gid)
            if 0.0 <= d <= self.max_range:
                ranges[i] = d
        return ranges


# ---------------------------------------------------------------------------
# tracker: lidar hits not explained by the known map -> remembered circles
# ---------------------------------------------------------------------------
def _dilate(occ, k):
    out = occ.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out |= np.roll(np.roll(occ, dy, 0), dx, 1)
    return out


def _enclosing(c1, c2):
    """Smallest circle containing circles c1, c2 = (x, y, r)."""
    d = float(np.hypot(c2[0] - c1[0], c2[1] - c1[1]))
    if d + c2[2] <= c1[2]:
        return list(c1)
    if d + c1[2] <= c2[2]:
        return list(c2)
    r = 0.5 * (d + c1[2] + c2[2])
    k = (r - c1[2]) / d
    return [c1[0] + (c2[0] - c1[0]) * k, c1[1] + (c2[1] - c1[1]) * k, r]


class ObstacleTracker:
    def __init__(self, gridmap, known_dilate=2, cluster_gap=0.3, min_points=2,
                 pad=0.2, shift=0.1, max_tracks=100):
        """pad   : extra radius added to every circle [m]
        shift : the lidar only sees the near face, so the cluster centre is pushed this far
                away from the sensor [m]. Together with `pad` this sets how much bigger than
                the real obstacle the planners think it is."""
        self.map = gridmap
        self.known = _dilate(gridmap.occ, known_dilate)
        self.cluster_gap, self.min_points, self.pad, self.max_tracks = cluster_gap, min_points, pad, max_tracks
        self.shift = shift
        self.circles = []  # [x, y, r]
        self.lock = threading.Lock()

    def _is_known(self, pts):
        m = self.map
        c = np.floor((pts[:, 0] - m.origin[0]) / m.resolution).astype(int)
        r = np.floor((pts[:, 1] - m.origin[1]) / m.resolution).astype(int)
        inb = (c >= 0) & (c < m.width) & (r >= 0) & (r < m.height)
        known = np.ones(len(pts), bool)  # outside the map -> ignore
        known[inb] = self.known[r[inb], c[inb]]
        return known

    def update(self, pose, ranges, rel_angles):
        idx = np.nonzero(np.isfinite(ranges))[0]
        if len(idx) == 0:
            return
        a = rel_angles[idx] + pose[2]
        pts = np.asarray(pose[:2]) + ranges[idx, None] * np.stack([np.cos(a), np.sin(a)], axis=1)
        keep = ~self._is_known(pts)
        idx, pts = idx[keep], pts[keep]
        if len(idx) == 0:
            return

        clusters, cur, last_i = [], [], None
        for i, p in zip(idx, pts):
            if cur and (i - last_i > 2 or np.hypot(*(p - cur[-1])) > self.cluster_gap):
                clusters.append(np.array(cur))
                cur = []
            cur.append(p)
            last_i = i
        if cur:
            clusters.append(np.array(cur))

        sensor = np.asarray(pose[:2])
        new = []
        for pc in clusters:
            if len(pc) < self.min_points:
                continue
            ctr = pc.mean(axis=0)
            away = ctr - sensor
            nrm = np.linalg.norm(away)
            if nrm > 1e-6:
                ctr = ctr + away / nrm * self.shift   # we only see the near face
            r = max(float(np.max(np.linalg.norm(pc - ctr, axis=1))), 0.1) + self.pad
            new.append([float(ctr[0]), float(ctr[1]), r])
        with self.lock:
            for c in new:
                self._merge(c)

    def _merge(self, c):
        for i, e in enumerate(self.circles):
            if np.hypot(c[0] - e[0], c[1] - e[1]) < c[2] + e[2]:
                self.circles[i] = _enclosing(e, c)
                return
        if len(self.circles) < self.max_tracks:
            self.circles.append(c)

    def count(self):
        with self.lock:
            return len(self.circles)

    def snapshot(self):
        with self.lock:
            return [list(c) for c in self.circles]

    def as_solver_rows(self, radius_offset=0.0):
        """(k,7) rows [x, y, a, b, theta, cos, sin] for TebplanSolver.
        radius_offset is added to every radius (negative = let TEB come closer, see ros_node)."""
        rows = [[x, y, max(r + radius_offset, 0.05), max(r + radius_offset, 0.05), 0.0, 1.0, 0.0]
                for x, y, r in self.snapshot()]
        return np.array(rows, float).reshape(-1, 7)

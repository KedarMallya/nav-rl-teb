"""Receding-horizon wrapper around TebplanSolver.

The controller is the only piece that talks to the planner, so it is the piece an
RL policy replaces later: keep the interface
    set_goal([x, y, yaw]);  update(pose, dt) -> (v, w, path_xy_or_None)
and swap the body of update().
"""
import threading
import numpy as np

from TebSolver import TebplanSolver


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TebController:
    def __init__(self, obstacles=None, global_planner=None, horizon=2.5, xy_tol=0.10,
                 yaw_tol=0.08, rot_thresh=1.0, **solver_kwargs):
        """
        obstacles   : (m,7) array [x, y, a, b, theta, cos, sin]
        global_planner : optional GlobalPlanner. If given, the local goal is taken along its
                      path (so walls are routed around); otherwise it is the straight-line
                      point towards the goal. Set `dynamic_circles` ((k,3) [x,y,r], e.g.
                      ObstacleTracker.snapshot()) from outside to make it replan around
                      newly sensed obstacles.
        horizon     : the solver plans to a local goal at most this far away [m]
        rot_thresh  : if the heading error to the local goal exceeds this [rad],
                      rotate in place first (the solver cannot reverse / U-turn)
        solver_kwargs: forwarded to TebplanSolver (v_max, omega_max, a_max, safe_distance, ...)
        """
        self.obstacles = np.empty((0, 7)) if obstacles is None else np.asarray(obstacles, float)
        self.horizon = horizon
        self.xy_tol = xy_tol
        self.yaw_tol = yaw_tol
        self.rot_thresh = rot_thresh
        solver_kwargs.setdefault("safe_distance", 0.5)
        self.solver_kwargs = solver_kwargs
        self.v_max = solver_kwargs.get("v_max", 1.0)
        self.w_max = solver_kwargs.get("omega_max", 1.0)
        self.a_max = solver_kwargs.get("a_max", 2.0)
        self.alpha_max = 3.0  # angular acceleration limit used for command smoothing [rad/s^2]

        self.gp = global_planner
        self.dynamic_circles = np.empty((0, 3))
        self.global_path = None     # (k,2) current global path, for visualisation / RL input

        self._solver = None
        self._goal = None
        self._lock = threading.Lock()
        self._cmd = np.zeros(2)
        self.done = True
        self._rotating = False  # hysteresis for the rotate-in-place phase
        self.last_status = "idle"

    # ------------------------------------------------------------------
    def set_goal(self, goal):
        with self._lock:
            self._goal = np.asarray(goal, dtype=float)
            self.done = False
            self._rotating = False
            self.global_path = None
            if self.gp is not None:
                self.gp.set_goal(self._goal)
            self.last_status = "new goal"

    def cancel(self):
        with self._lock:
            self._goal = None
            self.done = True
            self._cmd[:] = 0.0
            self.global_path = None
            if self.gp is not None:
                self.gp.clear()
            self.last_status = "idle"

    @property
    def goal(self):
        with self._lock:
            return None if self._goal is None else self._goal.copy()

    # ------------------------------------------------------------------
    def update(self, pose, dt, scan=None, vel=None):
        """pose=[x,y,yaw] in map frame, dt = time since the previous call. Returns (v, w, path).
        scan / vel are accepted for interface parity with RLController and ignored here."""
        goal = self.goal
        if goal is None or self.done:
            self._cmd[:] = 0.0
            return 0.0, 0.0, None
        pose = np.asarray(pose, dtype=float)

        # 0) global path (replans only when the goal, the sensed obstacles or the robot's
        #    position make the old one invalid)
        local = None
        if self.gp is not None:
            with self._lock:    # set_goal()/cancel() come from the ROS thread
                if self._goal is None:
                    return 0.0, 0.0, None
                goal = self._goal.copy()
                if not self.gp.update(pose[:2], self.dynamic_circles):
                    ok = False
                else:
                    ok = True
                    self.global_path = self.gp.path
                    goal = np.array([*self.gp.effective_goal, goal[2]])  # differs only if goal got blocked
                    local = self.gp.lookahead(pose[:2], self.horizon)
            if not ok:
                self.last_status = f"global planner: {self.gp.status}"
                self.global_path = None
                return (*self._slew(0.0, 0.0, dt), None)

        to_goal = goal[:2] - pose[:2]
        dist = float(np.linalg.norm(to_goal))

        # 1) position reached -> align with goal yaw, then finish
        if dist < self.xy_tol:
            err = wrap(goal[2] - pose[2])
            if abs(err) < self.yaw_tol:
                self.done = True
                self.last_status = "goal reached"
                self._cmd[:] = 0.0
                return 0.0, 0.0, None
            self.last_status = "final rotation"
            w = float(np.clip(2.0 * err, -self.w_max, self.w_max))
            return (*self._slew(0.0, w, dt), None)

        # 2) local goal (at most `horizon` away, heading toward the real goal)
        if local is not None:                       # along the global path
            xf_xy, tangent, _, is_final = local
            xf_yaw = goal[2] if is_final else tangent
            xf_yaw = pose[2] + wrap(xf_yaw - pose[2])
        elif dist <= self.horizon:
            xf_xy = goal[:2]
            xf_yaw = pose[2] + wrap(goal[2] - pose[2])
        else:
            xf_xy = pose[:2] + to_goal / dist * self.horizon
            xf_yaw = np.arctan2(to_goal[1], to_goal[0])
            xf_yaw = pose[2] + wrap(xf_yaw - pose[2])

        # 3) turn on the spot if we point the wrong way
        head_err = wrap(np.arctan2(xf_xy[1] - pose[1], xf_xy[0] - pose[0]) - pose[2])
        limit = 0.5 * self.rot_thresh if self._rotating else self.rot_thresh
        self._rotating = abs(head_err) > limit
        if self._rotating:
            self.last_status = "rotate to path"
            w = float(np.clip(2.0 * head_err, -self.w_max, self.w_max))
            return (*self._slew(0.0, w, dt), None)

        # 4) TEB
        x0 = np.array([pose[0], pose[1], pose[2]])
        xf = np.array([xf_xy[0], xf_xy[1], xf_yaw])
        obs = self._nearby_obstacles(pose[:2])
        try:
            if self._solver is None:
                self._solver = TebplanSolver(x0, xf, obs, **self.solver_kwargs)
            traj, dts = self._solver.solve(x0=x0, xf=xf, obstacles=obs)
        except Exception as exc:  # solver blew up -> stop rather than drive blind
            self.last_status = f"solver error: {exc}"
            self._cmd[:] = 0.0
            return 0.0, 0.0, None

        if not np.all(np.isfinite(traj)) or not np.all(np.isfinite(dts)):
            self.last_status = "non-finite plan"
            return (*self._slew(0.0, 0.0, dt), None)

        ok = bool(self._solver.solve_stats.get("success", False))
        self.last_status = "tracking" if ok else "tracking (solver not converged)"
        v, w = self._cmd_from_traj(traj, dts)
        return (*self._slew(v, w, dt), traj[:, :2])

    # ------------------------------------------------------------------
    def _nearby_obstacles(self, xy):
        if len(self.obstacles) == 0:
            return np.empty((0, 7))
        d = np.hypot(self.obstacles[:, 0] - xy[0], self.obstacles[:, 1] - xy[1])
        reach = self.horizon + 2.0 + np.max(self.obstacles[:, 2:4], axis=1)
        return self.obstacles[d < reach]

    def _cmd_from_traj(self, traj, dts):
        """Velocity of the first trajectory segment (what the robot must do right now)."""
        p0, p1 = traj[0], traj[1]
        dt0 = max(float(dts[0]), 1e-3)
        step = p1[:2] - p0[:2]
        v = np.linalg.norm(step) / dt0
        if np.dot(step, [np.cos(p0[2]), np.sin(p0[2])]) < 0:
            v = -v
        w = wrap(p1[2] - p0[2]) / dt0
        return float(np.clip(v, -self.v_max, self.v_max)), float(np.clip(w, -self.w_max, self.w_max))

    def _slew(self, v, w, dt):
        """Limit command change per step (a_max / alpha_max) so the sim sees smooth commands."""
        dt = max(dt, 1e-3)
        pv, pw = self._cmd
        v = float(np.clip(v, pv - self.a_max * dt, pv + self.a_max * dt))
        w = float(np.clip(w, pw - self.alpha_max * dt, pw + self.alpha_max * dt))
        self._cmd[:] = (v, w)
        return v, w

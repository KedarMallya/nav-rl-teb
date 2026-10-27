"""RL local controller: same interface as TebController, but the local planner is the trained
PPO policy (rl_controller.Policy) instead of the TEB optimiser.

    global A* path (GlobalPlanner)  ->  carrot point `lookahead` m ahead (line-of-sight)
        -> obs = [24 lidar beams, goal dist/bearing, v, w]  ->  PPO  ->  (v, w)

What the policy does NOT do, so this controller does it:
  * stop at the goal   - in training an episode ended at 0.25 m, the policy never learned to stop
  * align the goal yaw - the observation has no goal orientation
  * recover when stuck - it is purely reactive (24 beams, 3.5 m, no memory); after
                         `stuck_window_s` without progress the global path is recomputed,
                         `max_replans` times, then the goal is abandoned (self.failed = True)
Obstacles the lidar has passed are remembered by the tracker; the global planner replans
around them, the policy only ever sees the current scan.
"""
import threading
from collections import deque

import numpy as np

from rl_controller import RANGE, action_to_cmd, build_obs, scan_to_beams, to_local


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RLController:
    def __init__(self, global_planner, policy, lookahead=1.5, xy_tol=0.20, yaw_tol=0.08,
                 beam_window=0, range_offset=0.0, control_dt=0.1,
                 stuck_window_s=8.0, stuck_dist=0.10, max_replans=5,
                 a_max=2.0, alpha_max=3.0, w_max=1.0):
        """
        lookahead    : distance of the carrot along the global path [m] (training goals were
                       1.5-6.5 m away; the original Navigator used 1.0)
        xy_tol       : stop distance. Training success radius was 0.25 m.
        beam_window  : rays on each side of a beam centre that are min-ed. 0 = one ray per beam,
                       like the single MuJoCo rangefinders in training.
        range_offset : subtracted from every beam that sees something (< RANGE) [m]. Use it if
                       our robot's body is bigger than the training robot's, so the policy
                       thinks obstacles are closer than they are.
        control_dt   : the policy was trained at 10 Hz (DT_CONTROL = 0.1).
        a_max / alpha_max / w_max : only for the stop / final-rotation phases (the policy's own
                       commands are NOT slew-limited, like in training).
        """
        self.gp, self.policy = global_planner, policy
        self.lookahead, self.xy_tol, self.yaw_tol = lookahead, xy_tol, yaw_tol
        self.beam_window, self.range_offset = beam_window, range_offset
        self.a_max, self.alpha_max, self.w_max = a_max, alpha_max, w_max
        self.stuck_dist, self.max_replans = stuck_dist, max_replans
        self._hist = deque(maxlen=max(2, int(stuck_window_s / control_dt)))
        self._replans = 0

        self.obstacles = np.empty((0, 7))      # unused (the node assigns it for TEB)
        self.dynamic_circles = np.empty((0, 3))
        self.global_path = None
        self.local_goal = None
        self._goal = None
        self._lock = threading.Lock()
        self._cmd = np.zeros(2)
        self.done = True
        self.failed = False
        self.last_status = "idle"

    # ------------------------------------------------------------------
    def set_goal(self, goal):
        with self._lock:
            self._goal = np.asarray(goal, dtype=float)
            self.done, self.failed = False, False
            self._replans = 0
            self._hist.clear()
            self.global_path = self.local_goal = None
            self.gp.set_goal(self._goal)
            self.last_status = "new goal"

    def cancel(self):
        with self._lock:
            self._goal = None
            self.done, self.failed = True, False
            self._cmd[:] = 0.0
            self.global_path = self.local_goal = None
            self.gp.clear()
            self.last_status = "idle"

    @property
    def goal(self):
        with self._lock:
            return None if self._goal is None else self._goal.copy()

    # ------------------------------------------------------------------
    def update(self, pose, dt, scan=None, vel=None):
        """pose = [x, y, yaw] (map frame); dt = time since the last call;
        scan = (ranges, angle_min, angle_increment, range_min) of the latest lidar scan;
        vel = measured (v, w). Returns (v, w, path_for_display)."""
        goal = self.goal
        if goal is None or self.done:
            self._cmd[:] = 0.0
            return 0.0, 0.0, None
        pose = np.asarray(pose, dtype=float)

        # 0) global path (replans only when needed) and the carrot on it
        with self._lock:                      # set_goal()/cancel() come from the ROS thread
            if self._goal is None:
                return 0.0, 0.0, None
            goal = self._goal.copy()
            ok = self.gp.update(pose[:2], self.dynamic_circles)
            if ok:
                self.global_path = self.gp.path
                goal = np.array([*self.gp.effective_goal, goal[2]])   # differs only if blocked
                carrot, _, _, _ = self.gp.lookahead(pose[:2], self.lookahead)
        if not ok:
            self.last_status = f"global planner: {self.gp.status}"
            self.global_path = None
            return (*self._slew(0.0, 0.0, dt), None)

        # 1) position reached -> align with the goal yaw, then finish
        if np.linalg.norm(goal[:2] - pose[:2]) < self.xy_tol:
            err = wrap(goal[2] - pose[2])
            if abs(err) < self.yaw_tol:
                self.done = True
                self.last_status = "goal reached"
                self._cmd[:] = 0.0
                return 0.0, 0.0, None
            self.last_status = "final rotation"
            return (*self._slew(0.0, float(np.clip(2.0 * err, -self.w_max, self.w_max)), dt), None)

        if scan is None:
            self.last_status = "waiting for scan"
            return (*self._slew(0.0, 0.0, dt), None)

        # 2) stuck? (no progress for stuck_window_s) -> recompute the global path, a few times
        self._hist.append(pose[:2].copy())
        if len(self._hist) == self._hist.maxlen and \
                np.linalg.norm(self._hist[-1] - self._hist[0]) < self.stuck_dist:
            self._replans += 1
            self._hist.clear()
            if self._replans > self.max_replans:
                self.done = self.failed = True
                self._cmd[:] = 0.0
                self.last_status = "failed: stuck"
                return 0.0, 0.0, None
            with self._lock:
                self.gp.set_goal(self._goal)
            self.last_status = f"stuck: replanning ({self._replans}/{self.max_replans})"
            return (*self._slew(0.0, 0.0, dt), None)

        # 3) policy
        ranges, a_min, a_inc, r_min = scan
        beams = scan_to_beams(ranges, a_min, a_inc, r_min, window=self.beam_window)
        if self.range_offset:
            beams = np.where(beams < RANGE, np.maximum(beams - self.range_offset, 0.0), beams)
        v_meas, w_meas = (0.0, 0.0) if vel is None else vel
        obs = build_obs(beams, to_local(pose, carrot), v_meas, w_meas)
        v, w = (float(x) for x in action_to_cmd(self.policy(obs)))
        self._cmd[:] = (v, w)
        self.local_goal = carrot
        self.last_status = "policy" if getattr(self.policy, "model", True) is not None \
            else "policy (NO MODEL: obstacle-blind fallback)"
        return v, w, np.array([pose[:2], carrot])

    # ------------------------------------------------------------------
    def _slew(self, v, w, dt):
        dt = max(dt, 1e-3)
        pv, pw = self._cmd
        v = float(np.clip(v, pv - self.a_max * dt, pv + self.a_max * dt))
        w = float(np.clip(w, pw - self.alpha_max * dt, pw + self.alpha_max * dt))
        self._cmd[:] = (v, w)
        return v, w

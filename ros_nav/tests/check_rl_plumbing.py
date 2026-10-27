"""Run: python3 tests/check_rl_plumbing.py [n_seeds]
Checks the RL path WITHOUT mujoco / torch / stable-baselines3 (none of them needed):
  A) beam convention: our 360-ray scan -> 24 training beams (ahead=0, left=6, behind=12, right=18)
  B) closed loop through RLController with two stand-in policies
       blind : rl_controller.goal_seeking_policy (ignores the lidar)
       stub  : tiny hand-written reactive policy that only uses the beams -> proves the lidar
               reaches the policy with the right orientation / scaling
     physics = kinematic unicycle, lidar = numpy ray-caster, tracker + global planner = real code.
It does NOT test your trained model (load it with -p local_planner:=rl -p rl_model:=final.zip).
"""
import os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.modules["mujoco"] = types.ModuleType("mujoco")     # perception.py imports it for Lidar only
import numpy as np
from map_utils import load_map, grid_to_rects, distance_to_rects
from perception import ObstacleTracker, random_obstacles, extras_to_circles
from global_planner import GlobalPlanner
from rl_controller import (Policy, N_BEAMS, RANGE, goal_seeking_policy, scan_to_beams, build_obs,
                           action_to_cmd, to_local)
from rl_local_controller import RLController

m = load_map(os.path.join(ROOT, "maps", "demo.yaml"))
rects = grid_to_rects(m); ox, oy = m.origin[:2]; ex, ey = m.extent
ANG = np.linspace(-np.pi, np.pi, 360, endpoint=False)     # what perception.Lidar produces


def lidar(pose, truth, rng=4.0):
    a = ANG + pose[2]; r = np.arange(0.05, rng, 0.02)
    px = pose[0] + r[None] * np.cos(a)[:, None]; py = pose[1] + r[None] * np.sin(a)[:, None]
    c = np.floor((px - ox) / m.resolution).astype(int); rr = np.floor((py - oy) / m.resolution).astype(int)
    inb = (c >= 0) & (c < m.width) & (rr >= 0) & (rr < m.height)
    hit = np.zeros_like(px, bool); hit[inb] = m.occ[rr[inb], c[inb]]
    for x, y, rad in truth: hit |= np.hypot(px - x, py - y) < rad
    f = np.where(hit.any(1), hit.argmax(1), -1)
    return np.where(f >= 0, r[np.maximum(f, 0)], np.inf)


# ---------------------------------------------------------------- A) beam convention
ok = True
empty = np.empty((0, 3))
BASE = np.array([3.0, -4.0])      # open floor: nearest wall/pillar >= 1.8 m away
for yaw, off, expect in [(0, (1.2, 0), 0), (0, (0, 1.2), 6), (0, (0, -1.2), 18), (0, (-1.2, 0), 12),
                         (np.pi / 2, (0, 1.2), 0), (np.pi / 2, (-1.2, 0), 6), (-np.pi / 2, (0, -1.2), 0),
                         (np.pi, (0, 1.2), 18)]:
    pose = np.array([*BASE, yaw])
    r = lidar(pose, np.array([[BASE[0] + off[0], BASE[1] + off[1], 0.3]]))
    b = scan_to_beams(r, ANG[0], 2 * np.pi / 360, 0.05, window=0)
    got = int(np.argmin(b)); good = got == expect and abs(b[got] - 0.9) < 0.05
    ok &= good
    print(f"  yaw={yaw:+.2f} obstacle {off} from robot: nearest beam {got} (want {expect}), range {b[got]:.2f} (want 0.90)  {'ok' if good else 'WRONG'}")
obs = build_obs(np.full(N_BEAMS, 2.0), to_local((0, 0, 0), (1.0, 1.0)), 0.13, -0.91)
print("  obs shape", obs.shape, "dtype", obs.dtype, "range", obs.min().round(2), obs.max().round(2),
      "tail", np.round(obs[-5:], 3))
ok &= obs.shape == (29,)
print("A) beam convention:", "OK" if ok else "FAILED")


# ---------------------------------------------------------------- B) closed loop
def reactive_stub(o):
    """goal seeking + steer away from beams closer than 0.6 m in the front half (beams only)."""
    a = goal_seeking_policy(o).astype(float)
    d = o[:N_BEAMS] * RANGE; ang = 2 * np.pi * np.arange(N_BEAMS) / N_BEAMS
    close = np.clip(1.0 - d / 0.6, 0, None) * (np.cos(ang) > 0.2)
    a[1] -= 1.5 * float(np.sum(close * np.sin(ang)))                     # obstacle on the left -> turn right
    if d[[0, 1, 23]].min() < 0.45: a[0] = -0.5                           # blocked ahead: slow right down
    return np.clip(a, -1, 1).astype(np.float32)


class Wrap:
    def __init__(self, f): self.f, self.model = f, "stub"
    def __call__(self, o): return self.f(o)


def run(seed, policy, goals, timeout=150.0):
    extras = random_obstacles(rects, (ox + .6, oy + .6, ox + ex - .6, oy + ey - .6), 8, seed, keep_out=[(0, 0, .25)])
    truth = extras_to_circles(extras)
    tracker = ObstacleTracker(m, pad=0.08, shift=0.15)
    ctrl = RLController(GlobalPlanner(m), policy)
    pose = np.array([0., 0., 0.]); v = w = 0.0; DT = 0.1; hits = 0; res = []
    for g in goals:
        ctrl.set_goal(g); t = 0.0
        while not ctrl.done and t < timeout:
            ranges = lidar(pose, truth); tracker.update(pose, ranges, ANG)
            ctrl.dynamic_circles = np.array(tracker.snapshot(), float).reshape(-1, 3)
            v, w, _ = ctrl.update(pose, DT, scan=(ranges, ANG[0], 2 * np.pi / 360, 0.05), vel=(v, w))
            pose = pose + DT * np.array([v * np.cos(pose[2]), v * np.sin(pose[2]), w]); t += DT
            if distance_to_rects(pose[:2], rects) < 0.22 or np.any(np.hypot(truth[:, 0] - pose[0], truth[:, 1] - pose[1]) < truth[:, 2] + 0.22):
                hits += 1
        res.append((ctrl.done and not ctrl.failed, t, ctrl.last_status))
    return hits, res


goals = [(5, 0, 0), (5, 3, np.pi / 2), (-2, 3.5, np.pi), (-4, -4, 0), (4.5, 4.5, -np.pi / 2)]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
print("B) closed loop, %d seeds x %d goals (global path + lidar tracker are the real code)" % (n, len(goals)))
for name, pol in [("blind (goal-seeking)", Policy(None)), ("stub reactive      ", Wrap(reactive_stub))]:
    tot_hits = tot_reached = tot_t = 0
    for s in range(1, n + 1):
        h, res = run(s, pol, goals)
        tot_hits += h; tot_reached += sum(r[0] for r in res); tot_t += sum(r[1] for r in res)
    print(f"  {name}: goals reached {tot_reached}/{n*len(goals)}, collision steps {tot_hits}, mean time/goal {tot_t/(n*len(goals)):.0f}s")

"""Run: python3 tests/check_closed_loop.py [seed] [old|new]   (old = wide margins, new = tight margins)
Closed loop WITHOUT mujoco/casadi: real controller + global planner + tracker + map utils;
fake TEB solver (straight line to local goal), kinematic unicycle, numpy lidar."""
import sys, types, time
import os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np

sys.modules["mujoco"] = types.ModuleType("mujoco")          # perception.py only needs it for Lidar
fake = types.ModuleType("TebSolver")
class FakeSolver:
    def __init__(self, x0, xf, obs=None, **kw): self.solve_stats = {"success": True}
    def solve(self, x0=None, xf=None, obstacles=None):
        """Stand-in for a nonholonomic local planner: straight line to the local goal, first
        segment turns towards it (w = 3*err) and slows down while misaligned."""
        x0, xf = np.asarray(x0, float), np.asarray(xf, float)
        n = 8; t = np.linspace(0, 1, n + 2)[:, None]
        traj = x0 + t * (xf - x0); d = np.linalg.norm(xf[:2] - x0[:2])
        dt = max(d / (n + 1) / 0.8, 0.05)
        err = (np.arctan2(*(xf[:2] - x0[:2])[::-1]) - x0[2] + np.pi) % (2 * np.pi) - np.pi
        step = traj[1, :2] - traj[0, :2]
        traj[1, :2] = x0[:2] + step * max(0.0, np.cos(err)) ** 2
        traj[1, 2] = x0[2] + 3.0 * err * dt
        return traj, np.full(n + 1, dt)
fake.TebplanSolver = FakeSolver; sys.modules["TebSolver"] = fake

from map_utils import load_map, grid_to_rects, rects_to_ellipses, distance_to_rects
from perception import ObstacleTracker, random_obstacles, extras_to_circles
from global_planner import GlobalPlanner
from teb_controller import TebController
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
m = load_map(os.path.join(ROOT, "maps", "demo.yaml"))
rects = grid_to_rects(m); ell = rects_to_ellipses(rects)
ex, ey = m.extent; ox, oy = m.origin[:2]
extras = random_obstacles(rects, (ox+.6, oy+.6, ox+ex-.6, oy+ey-.6), 8, seed, keep_out=[(0, 0, .25)])
truth = extras_to_circles(extras)
MODE = sys.argv[2] if len(sys.argv) > 2 else "new"
if MODE == "old":
    tracker = ObstacleTracker(m, pad=0.20, shift=0.10); gp = GlobalPlanner(m, dyn_near=0.50, dyn_far=0.70)
else:
    tracker = ObstacleTracker(m, pad=0.08, shift=0.15); gp = GlobalPlanner(m)
ctrl = TebController(ell, global_planner=gp)
R = 0.25

def lidar(pose, n=360, rng=4.0):
    ang = np.linspace(-np.pi, np.pi, n, endpoint=False) + pose[2]
    r = np.arange(0.05, rng, 0.02)
    px = pose[0] + r[None, :] * np.cos(ang)[:, None]; py = pose[1] + r[None, :] * np.sin(ang)[:, None]
    c = np.floor((px - ox) / m.resolution).astype(int); rr = np.floor((py - oy) / m.resolution).astype(int)
    inb = (c >= 0) & (c < m.width) & (rr >= 0) & (rr < m.height)
    hit = np.zeros_like(px, bool); hit[inb] = m.occ[rr[inb], c[inb]]
    for x, y, rad in truth: hit |= np.hypot(px - x, py - y) < rad
    first = np.where(hit.any(1), hit.argmax(1), -1)
    return np.where(first >= 0, r[np.maximum(first, 0)], np.inf), ang - pose[2]

def collides(p):
    return distance_to_rects(p[:2], rects) < R * 0.9 or (len(truth) and np.any(np.hypot(truth[:, 0]-p[0], truth[:, 1]-p[1]) < truth[:, 2] + R * 0.9))

pose = np.array([0., 0., 0.]); v = w = 0.0; DT = 0.1
goals = [(5, 0, 0), (5, 3, np.pi/2), (-2, 3.5, np.pi), (-4, -4, 0), (2, 4.5, 0), (4.5, 4.5, -np.pi/2), (0, 0, 0)]
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(m.occ, origin="lower", extent=(ox, ox+ex, oy, oy+ey), cmap="Greys")
for x, y, r in truth: ax.add_patch(plt.Circle((x, y), r, color="m", alpha=.4))
all_ok = True
for gi, g in enumerate(goals):
    ctrl.set_goal(g); trail = []; t0 = time.time(); hits = 0; simt = 0.0; plans0 = gp.version
    while not ctrl.done and simt < 90:
        ranges, rel = lidar(pose); tracker.update(pose, ranges, rel)
        ctrl.dynamic_circles = np.array(tracker.snapshot(), float).reshape(-1, 3)
        sensed = tracker.as_solver_rows()
        ctrl.obstacles = np.vstack([ell, sensed]) if len(sensed) else ell
        v, w, _ = ctrl.update(pose, DT)
        pose = pose + DT * np.array([v*np.cos(pose[2]), v*np.sin(pose[2]), w]); trail.append(pose.copy()); simt += DT
        hits += collides(pose)
        if ctrl.last_status.startswith("global planner"): break
    trail = np.array(trail); err = np.hypot(*(np.array(g[:2]) - pose[:2]))
    ok = ctrl.done and hits == 0; all_ok &= ok
    print(f"goal {gi} {tuple(np.round(g,2))}: reached={ctrl.done} err={err:.2f} collision_steps={hits} "
          f"sim_t={simt:.1f}s replans={gp.version-plans0} status={ctrl.last_status!r} wall={time.time()-t0:.1f}s")
    ax.plot(trail[:, 0], trail[:, 1], lw=1.5, label=f"goal {gi}"); ax.plot(*g[:2], "g*")
for x, y, r in tracker.snapshot(): ax.add_patch(plt.Circle((x, y), r, fill=False, color="c"))
ax.legend(fontsize=7); ax.set_title(f"seed {seed}: {len(truth)} hidden, {tracker.count()} sensed")
fig.savefig(os.path.join(HERE, f"loop_seed{seed}.png"), dpi=90)
print("ALL OK" if all_ok else "SOME FAILED", "| hidden", len(truth), "sensed", tracker.count())

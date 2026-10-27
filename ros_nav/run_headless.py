#!/usr/bin/env python3
"""ROS-free closed-loop check: goals -> TEB -> MuJoCo. Prints a summary, saves a plot.

    python3 run_headless.py            # default goal list
    python3 run_headless.py 4 2 1.57   # single goal x y yaw
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from robot_sim import DiffDriveSim, make_obstacles
from teb_controller import TebController

START = (0.0, 0.0, 0.0)
OBSTACLES = make_obstacles([
    (2.0, 0.3, 0.30, 0.30, 0.0),
    (3.5, -1.0, 0.50, 0.25, 0.6),
    (4.5, 1.5, 0.30, 0.30, 0.0),
    (-1.5, 2.0, 0.40, 0.25, 1.2),
])
DEFAULT_GOALS = [(5.0, 0.0, 0.0), (5.0, 3.0, np.pi / 2), (-2.0, 3.5, np.pi)]
PLAN_DT = 0.1
TIMEOUT = 60.0  # sim seconds per goal


def run_goal(sim, ctrl, goal):
    ctrl.set_goal(goal)
    trail, t_start, collisions = [], sim.time, 0
    while not ctrl.done and sim.time - t_start < TIMEOUT:
        v, w, _ = ctrl.update(sim.pose, PLAN_DT)
        sim.set_cmd(v, w)
        sim.step(PLAN_DT)
        trail.append(sim.pose.copy())
        collisions += sim.obstacle_contacts() > 0
    err = np.hypot(*(np.array(goal[:2]) - sim.pose[:2]))
    return np.array(trail), ctrl.done, err, collisions, sim.time - t_start


def main():
    goals = [tuple(map(float, sys.argv[1:4]))] if len(sys.argv) == 4 else DEFAULT_GOALS
    sim = DiffDriveSim(START, OBSTACLES)
    ctrl = TebController(OBSTACLES)

    fig, ax = plt.subplots(figsize=(7, 6))
    for x, y, a, b, th, *_ in OBSTACLES:
        ax.add_patch(Ellipse((x, y), 2 * a, 2 * b, angle=np.degrees(th), color="tab:red", alpha=0.6))
    all_ok = True
    for i, g in enumerate(goals):
        trail, reached, err, hits, dur = run_goal(sim, ctrl, g)
        all_ok &= reached and hits == 0
        print(f"goal {i}: {g}  reached={reached}  pos_err={err:.3f} m  "
              f"obstacle_contact_steps={hits}  time={dur:.1f}s  status={ctrl.last_status}")
        if len(trail):
            ax.plot(trail[:, 0], trail[:, 1], label=f"goal {i}")
        ax.annotate("", xy=(g[0] + 0.3 * np.cos(g[2]), g[1] + 0.3 * np.sin(g[2])), xytext=g[:2],
                    arrowprops=dict(arrowstyle="->", color="green"))
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
    fig.savefig("headless_run.png", dpi=120)
    print("plot -> headless_run.png;", "ALL OK" if all_ok else "SOME GOALS FAILED")


if __name__ == "__main__":
    main()

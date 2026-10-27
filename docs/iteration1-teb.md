# Iteration 1: TEB navigation stack on MuJoCo

> This document describes the project as of tag `v1-teb`. The code now lives in `ros_nav/`. Later changes are described in [iteration3-rl.md](iteration3-rl.md).

A small navigation stack for a differential-drive robot. You click a goal in RViz, a global planner finds a path through the map, and a TEB (Timed Elastic Band) local planner drives the robot along it inside a MuJoCo simulation. Hidden obstacles that are not on the map are detected with a simulated lidar and avoided.

This is the baseline. Iteration 2 trains an RL policy and iteration 3 swaps it in for the TEB controller.

```
RViz "2D Goal Pose"
        |
        v
   ros_node.py  --->  GlobalPlanner (A* on the map + sensed obstacles)
        |                      |
        |                      v  local goal ("lookahead" point on the path)
        |               TebController  (TebSolver, CasADi / IPOPT)
        |                      |
        v                      v
  MuJoCo diff-drive robot  <-- (v, w)
        |
        +--> /odom, TF, /scan, /map, paths and markers back to RViz
```

## What is in here

| File | What it does |
|---|---|
| `ros_node.py` | The ROS 2 node. Runs the simulation, publishes `/odom`, `/scan`, `/map`, TF and the RViz markers, receives `/goal_pose` and runs the planner thread. |
| `map_utils.py` | Loads a ROS `map_server` map (YAML + PGM) and turns it into rectangles. The same rectangles feed RViz, MuJoCo and the planners, so they cannot disagree. Also generates the demo map. |
| `global_planner.py` | A* on the occupancy grid, with extra cost near obstacles. It also plans around the circles found by the lidar tracker and hands out the local goal. |
| `perception.py` | Random hidden obstacles, the simulated 360 ray lidar, and `ObstacleTracker`, which turns lidar hits the map cannot explain into remembered circles. |
| `robot_sim.py` | Builds the MuJoCo world and the robot, and steps the physics. |
| `teb_controller.py` | Receding-horizon wrapper around the solver. Rotates in place when misaligned, tracks the local goal, aligns to the goal yaw at the end. |
| `TebSolver.py` | The TEB optimiser (CasADi + IPOPT). |
| `teb.rviz` | RViz layout. |
| `maps/demo.yaml`, `maps/demo.pgm` | The demo map, a 12 x 12 m arena with a half wall, a wall with a gap and four pillars. |
| `run_headless.py` | Runs the controller in MuJoCo without ROS and saves a plot. |
| `tests/` | Offline checks (see below). |

`TebSolver.py` is based on [whsleep/Minimal_TEB](https://github.com/whsleep/Minimal_TEB) with a few local changes: empty obstacle lists are handled, the IPOPT output is silenced and the solver statistics are kept so the caller can check for convergence. Check that repository's licence before publishing.

## Setup

Tested on Ubuntu (WSL2 on Windows 11) with ROS 2. Use whichever ROS 2 distro you have installed.

```bash
source /opt/ros/<distro>/setup.bash
pip install mujoco numpy pyyaml pillow casadi matplotlib
```

If you use a virtual environment, create it with `python3 -m venv --system-site-packages venv` so that it can still see `rclpy`.

The two check scripts `check_closed_loop.py` and `check_global_planner.py` have to sit in a `tests/` folder next to the code, because they look for `maps/demo.yaml` one folder above themselves.

### Showing the MuJoCo window under WSL2

Windows 11 with WSLg usually works with:

```bash
export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
```

If the window still does not open, run with `-p viewer:=false` and watch the robot in RViz instead.

## Running

Start the node:

```bash
python3 ros_node.py
```

In a second terminal start RViz with the supplied layout:

```bash
source /opt/ros/<distro>/setup.bash
rviz2 -d teb.rviz
```

Click **2D Goal Pose** and click a free spot on the map. The robot plans a path and drives there.

If you build your own RViz setup, set Fixed Frame to `map` and add a Map display on `/map` with Durability set to Transient Local.

### What you see in RViz

| Topic | Meaning |
|---|---|
| `/map` | The map the planner knows about |
| `/global_path` | A* path |
| `/teb_path` | The current optimised local trajectory |
| `/teb_obstacles` | The planner's view of the map obstacles (orange) |
| `/teb_sensed` | Hidden obstacles found by the lidar so far (cyan) |
| `/teb_truth` | The real hidden obstacles (magenta, only with `show_truth:=true`) |
| `/scan` | Simulated lidar |

### Parameters

Pass them with `--ros-args -p name:=value`.

| Parameter | Default | Meaning |
|---|---|---|
| `map_yaml` | `maps/demo.yaml` | Map to load. The demo map is created if the default is missing. |
| `rect_max_len` | 1.0 | Largest side of the rectangles the map is split into [m] |
| `n_random` | 8 | Number of hidden random obstacles |
| `seed` | -1 | Layout seed. -1 gives a new layout every run. Reuse a seed to repeat a run. |
| `show_truth` | false | Draw the real hidden obstacles (debugging only) |
| `lidar_range`, `scan_rate` | 4.0, 10.0 | Lidar range [m] and rate [Hz] |
| `sensed_pad`, `sensed_shift` | 0.08, 0.15 | How the tracker inflates and shifts a detected obstacle [m] |
| `sensed_clearance` | 0.30 | Distance TEB keeps from a sensed obstacle [m] |
| `start_x`, `start_y`, `start_yaw` | 0.0 | Start pose (write them with a decimal point) |
| `viewer` | true | Open the MuJoCo window |
| `goal_topic`, `frame` | `/goal_pose`, `map` | Goal topic and the fixed frame |

If the robot clips hidden obstacles, raise `sensed_pad` or `sensed_clearance`. To squeeze closer to them, lower the same values.

## Checks without ROS

These run on plain Python and are a quick way to see that a change did not break anything.

```bash
python3 tests/check_global_planner.py     # A* on the demo map: path length, clearance, planning time
python3 tests/check_closed_loop.py 1 new  # whole loop with a fake solver and a numpy lidar (no MuJoCo, no CasADi)
python3 run_headless.py                   # TEB in MuJoCo without ROS, saves a plot
```

## How the pieces work together

1. The map is split into rectangles. MuJoCo gets them as boxes, TEB gets them as circumscribing ellipses and RViz gets the occupancy grid.
2. Random obstacles exist only inside MuJoCo. The lidar sees them, the tracker clusters the unexplained hits into circles and remembers them, and the global planner treats those circles like occupied cells.
3. The global planner replans only when the goal, the sensed obstacles or the robot position make the old path invalid.
4. `TebController` asks the planner for a point about 2.5 m along the path, rotates in place if it points the wrong way, runs the TEB solver towards that point and applies the first step of the trajectory, limited by acceleration.
5. At the goal it rotates to the requested yaw and stops.

`TebController.update(pose, dt)` is the interface an RL policy replaces later: `set_goal([x, y, yaw])`, then `update(...)` returning `(v, w, path)`.

## Known limitations

- The TEB solver cannot reverse or make a U-turn, which is why the controller turns on the spot first.
- Obstacles are assumed to be static once seen.
- In `robot_sim.py` both casters sit on the robot's centreline, so the robot can rest on three contacts and lift one drive wheel. It then moves slower than commanded. Moving the casters up by 2 mm (`z = -0.028` instead of `-0.03` for `caster_f` and `caster_r`) fixes it. This was found after this iteration was frozen, so the code at tag `v1-teb` is left as it was. The fix is applied in `v3-rl`.

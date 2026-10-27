# Iteration 3: RL local planner in the ROS 2 / MuJoCo navigation stack

> This document describes the project as of tag `v3-rl`. The stack lives in `ros_nav/` and the training code in `training/`. The baseline it builds on is described in [iteration1-teb.md](iteration1-teb.md).

This is the final stage of the project. The navigation stack from iteration 1 (RViz goals, A* global planner, lidar obstacle tracker, MuJoCo robot) is unchanged, but the TEB local planner can now be replaced by a policy trained with reinforcement learning (PPO from Stable-Baselines3, trained in iteration 2). A single ROS parameter chooses which one drives the robot.

```
RViz "2D Goal Pose"
        |
        v
   ros_node.py --> GlobalPlanner (A*) --> carrot point on the path
        |                                       |
        |                    local_planner:=teb |  local_planner:=rl
        |                                       v
        |                      TebController   or   RLController
        |                      (TEB optimiser)       (PPO policy + stop / align / recovery)
        v                                       |
  MuJoCo diff-drive robot  <----------------  (v, w)
```

Both controllers have the same interface, `set_goal([x, y, yaw])` and `update(pose, dt, scan, vel)` returning `(v, w, path)`, so the rest of the stack does not know which one is running.

## Contents

1. [Layout](#layout)
2. [The RL algorithm](#the-rl-algorithm)
3. [Training the policy](#training-the-policy)
4. [Running the final pipeline](#running-the-final-pipeline)
5. [How the RL controller works](#how-the-rl-controller-works)
6. [Tuning](#tuning)
7. [Checks](#checks)
8. [Changes from iteration 1](#changes-from-iteration-1)
9. [Known limitations](#known-limitations)

## Layout

The stack in `ros_nav/` is the iteration 1 code plus the RL controller. See [iteration1-teb.md](iteration1-teb.md) for the files that did not change (`map_utils.py`, `global_planner.py`, `perception.py`, `robot_sim.py`, `TebSolver.py`, `teb.rviz`, `maps/`).

| File | Role |
|---|---|
| `ros_node.py` | ROS 2 node. New parameters choose and configure the local planner. |
| `teb_controller.py` | TEB controller, now accepting (and ignoring) `scan` and `vel` so both controllers share an interface. |
| `rl_controller.py` | Observation building, lidar to 24 beams conversion, action scaling and the `Policy` wrapper that loads the PPO model. |
| `rl_local_controller.py` | `RLController`. Runs the policy and adds what the policy cannot do itself: stopping, goal yaw alignment and stuck recovery. |
| `final.zip` | The trained PPO model. |
| `tests/check_rl_plumbing.py` | Offline check of the RL path that needs no MuJoCo, PyTorch or Stable-Baselines3. |
| `tests/check_closed_loop.py`, `tests/check_global_planner.py` | Offline checks inherited from iteration 1. |

The training code belongs to iteration 2 and lives in `training/`:

| File | Role |
|---|---|
| `nav_min_mujoco.py` | Gymnasium environment, reward, PPO training, baseline and evaluation commands. |
| `models/mobile_robot.xml` | MuJoCo model of the training robot and room. |
| `rl_constants.py` | The five numbers that training and ROS must agree on. |
| `rl_controller.py` | The same observation and action code used in `ros_nav/`. |

## The RL algorithm

The policy is a small neural network trained with **PPO** (Proximal Policy Optimization) from [Stable-Baselines3](https://stable-baselines3.readthedocs.io). It is purely reactive: it sees the current lidar scan and a goal direction and outputs a speed command. It has no map and no memory.

### Environment

A 6 x 6 m room with four walls and up to five box obstacles (0.44 m wide) at random places. Each episode starts with a random start pose, a random heading and a random goal at least 1.5 m away. The simulation runs at 0.002 s and the policy acts at 10 Hz.

### Observation (29 numbers)

| Index | Value | Scaling |
|---|---|---|
| 0 to 23 | 24 lidar beams. Beam 0 points straight ahead, the rest go counter-clockwise (beam 6 is left, 12 behind, 18 right). | range / 3.5 m, no return counts as 3.5 m |
| 24 | Distance to goal | distance / 6 m, capped at 1 |
| 25, 26 | Goal bearing in the robot frame | sin and cos |
| 27 | Forward speed | v / 0.26 |
| 28 | Turn rate | w / 1.82 |

### Action (2 numbers in [-1, 1])

| Index | Meaning |
|---|---|
| 0 | Forward speed: `v = 0.5 * (a0 + 1) * 0.26 m/s`. The robot cannot reverse. |
| 1 | Turn rate: `w = a1 * 1.82 rad/s` |

The command is converted to left and right wheel speeds for a differential drive.

### Reward

| Term | Value |
|---|---|
| Progress towards the goal | +5 per metre gained |
| Time penalty | -0.01 per step |
| Reaching the goal (within 0.25 m) | +10, episode ends |
| Any collision (wall or obstacle) | -10, episode ends |
| 400 steps (40 s) without either | Episode is cut off (truncated, not terminated) |

### PPO settings

| Setting | Value |
|---|---|
| Policy | MLP, two hidden layers of 64 |
| Parallel environments | 4 |
| Steps per environment per update | 512 |
| Batch size, epochs per update | 256, 8 |
| Learning rate, discount | 3e-4, 0.99 |
| Entropy coefficient | 0.005 |
| Lidar noise in training | 0.01 m |
| Default training length | 500,000 steps |

### Curriculum

The number of obstacles is raised from 0 to 5 over the first half of training, so the policy learns to reach goals first and to avoid obstacles after that. Two extra callbacks log `rollout/success_rate` and `rollout/collision_rate` to TensorBoard and a checkpoint is saved every 50,000 steps.

## Training the policy

Training runs on the CPU and needs no display, so it works fine in WSL2. Keep the project in the Linux file system (`~/...`) and not under `/mnt/c`, which is much slower. Run these commands from the `training/` folder.

```bash
python3 -m venv venv && source venv/bin/activate
pip install mujoco gymnasium stable-baselines3 tensorboard
pip install torch --index-url https://download.pytorch.org/whl/cpu    # if not already installed

python nav_min_mujoco.py selftest                     # environment checks
python nav_min_mujoco.py baseline                     # obstacle-blind goal seeker, for reference
python nav_min_mujoco.py train --steps 500000         # writes runs/min_mujoco/final.zip
python nav_min_mujoco.py eval --model runs/min_mujoco/final.zip
tensorboard --logdir runs/min_mujoco/tb               # in a second terminal
```

`eval` runs 100 episodes each with 0, 3 and 5 obstacles and prints the share of successes, collisions and timeouts. The trained policy should clearly beat the baseline at 3 and 5 obstacles. Add `--render` to watch it if your system has a display. To continue an interrupted run use `--resume runs/min_mujoco/ckpt/<file>.zip`.

The `runs/` folder is not kept in git. The trained model that the ROS stack uses is committed as `ros_nav/final.zip`.

## Running the final pipeline

Run these commands from the `ros_nav/` folder, where `final.zip` is. Install the dependencies:

```bash
source /opt/ros/<distro>/setup.bash
pip install mujoco numpy pyyaml pillow casadi matplotlib stable-baselines3
```

`casadi` is still needed with the RL planner, because `ros_node.py` imports the TEB controller when it starts. If you use a virtual environment, create it with `--system-site-packages` so that `rclpy` is visible.

Start the node with the RL planner:

```bash
python3 ros_node.py --ros-args -p local_planner:=rl -p rl_model:=final.zip -p seed:=7 -p rl_range_offset:=0.08
```

Then open RViz in a second terminal and click **2D Goal Pose**:

```bash
source /opt/ros/<distro>/setup.bash
rviz2 -d teb.rviz
```

To compare with TEB on exactly the same hidden obstacles, run the same command with the same seed and `local_planner:=teb`.

If the log says `NO MODEL: obstacle-blind fallback`, the path given in `rl_model` was not found and the node is running a simple goal-seeking rule that ignores the lidar.

### Showing the MuJoCo window under WSL2

```bash
export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
sudo apt install -y x11-apps mesa-utils libglfw3 libgl1-mesa-dri libegl1
xeyes        # a window should appear on the Windows desktop
```

If the window still does not open, add `-p viewer:=false` and watch the robot in RViz.

### Parameters

All the iteration 1 parameters still apply (see [iteration1-teb.md](iteration1-teb.md)). The new ones are:

| Parameter | Default | Meaning |
|---|---|---|
| `local_planner` | `teb` | `teb` or `rl` |
| `rl_model` | empty | Path to the PPO `.zip`. Empty falls back to the obstacle-blind rule. |
| `rl_lookahead` | 1.5 | How far ahead on the global path the policy's goal point is [m] |
| `rl_range_offset` | 0.0 | Subtracted from every lidar beam that sees something [m], so the policy believes obstacles are closer than they are |

`sensed_clearance` only affects TEB.

## How the RL controller works

`RLController` does the following on every planning step (10 Hz):

1. Gets the current global path and picks a **carrot**, the point `rl_lookahead` metres ahead on it.
2. If the robot is within 0.20 m of the goal, it rotates in place to the requested yaw and finishes. The policy never learned this, because training episodes ended at 0.25 m and the observation has no goal orientation.
3. Checks whether the robot is **stuck**. If it moved less than 0.10 m in 8 s, the global path is recomputed. After 5 such recoveries the goal is abandoned and `failed` is set.
4. Resamples the 360 ray scan to the 24 training beams, builds the 29 number observation with the carrot as the goal, and asks the policy for `(v, w)`.

The policy only ever sees the current scan. Obstacles it has already passed are remembered by the lidar tracker, and the global planner routes around them, so the carrot steers the robot clear of them.

## Tuning

| Symptom | Try |
|---|---|
| Clips obstacles or walls | Raise `rl_range_offset` (start at 0.08, go up in steps of 0.03). The ROS robot is bigger than the training robot. |
| Refuses to pass through reasonable gaps | Lower `rl_range_offset`. |
| Cuts corners or hesitates | Change `rl_lookahead` between 1.0 and 2.0. |
| Spins in place in front of an obstacle | Look at `planner:` messages in the log. Repeated `stuck: replanning` means the policy is trapped and the global path needs to go around. |

## Checks

From the `ros_nav/` folder:

```bash
python3 tests/check_rl_plumbing.py 4
```

Part A confirms that the 360 ray lidar is converted to the right training beams (ahead is beam 0, left is 6, behind is 12, right is 18) for several robot headings, and that the observation has 29 numbers. Part B runs the real global planner and tracker in a closed loop with two stand-in policies. It does **not** test your trained model, only the wiring around it.

The two constants lists must stay identical in training and in ROS: `N_BEAMS = 24`, `RANGE = 3.5`, `ROOM = 6.0`, `V_MAX = 0.26`, `W_MAX = 1.82`. If you change any of them, retrain.

## Changes from iteration 1

- `ros_node.py`: added `local_planner`, `rl_model`, `rl_lookahead` and `rl_range_offset`, builds the chosen controller, records the latest lidar scan and measured velocity and passes them to `update()`.
- `teb_controller.py`: `update()` accepts `scan` and `vel` and ignores them.
- New: `rl_controller.py`, `rl_local_controller.py`, `tests/check_rl_plumbing.py`.
- `robot_sim.py`: the casters are raised by 2 mm (`z = -0.028` for `caster_f` and `caster_r`). With the original value one drive wheel could lift off the floor and the robot moved at only about a third of the commanded speed. After the change it reaches 0.26 m/s and 1.77 rad/s at full command, which is what the policy was trained on.

## Known limitations

- The policy was trained in an empty square room with a few boxes. It has never seen walls, corridors or doorways, so the global path carries most of the navigation and the policy mainly handles local avoidance.
- The training robot (cylinder, radius 0.16 m, track 0.26 m) is smaller than the ROS robot (0.40 x 0.28 m box, track 0.32 m). `rl_range_offset` compensates for it, and training on the larger robot would be the proper fix.
- The policy cannot reverse and has no memory, so it can get stuck in dead ends. The stuck recovery replans the global path but cannot back out.
- The maximum speed is 0.26 m/s, so long goals take a while compared to TEB, which is allowed to go faster.
- This has been tested offline and in the simulator. It has not been run on a real robot.

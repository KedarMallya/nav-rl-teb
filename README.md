# nav-rl-teb: TEB vs. reinforcement learning local planning in ROS 2 and MuJoCo

A small navigation stack for a differential-drive robot, built in three stages. You click a goal in RViz, an A* global planner finds a path, and a local planner drives the robot along it inside a MuJoCo simulation. Hidden obstacles that are not on the map are found with a simulated lidar and avoided.

The local planner is swappable. You can run a classical **TEB** (Timed Elastic Band) optimiser or a **PPO policy** trained with reinforcement learning, switch between them with one ROS parameter and compare both on the same hidden obstacles.

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

## Project stages

| Stage | What it adds | Tag | Docs |
|---|---|---|---|
| 1. Baseline | ROS 2 node, map handling, A* planner, lidar obstacle tracker, MuJoCo robot, TEB local planner | `v1-teb` | [docs/iteration1-teb.md](docs/iteration1-teb.md) |
| 2. Training | Gymnasium environment in MuJoCo, PPO training, baseline and evaluation commands | `v2-training` | [docs/iteration3-rl.md](docs/iteration3-rl.md#training-the-policy) |
| 3. Integration | RL controller in the ROS stack, selectable with `local_planner:=rl` | `v3-rl` | [docs/iteration3-rl.md](docs/iteration3-rl.md) |

Each stage is a tag, so you can check out the exact state of any stage, for example `git checkout v1-teb` for the TEB baseline.

## Repository layout

```
nav-rl-teb/
├── ros_nav/        ROS 2 / MuJoCo stack (stages 1 and 3)
│   ├── ros_node.py, map_utils.py, global_planner.py, perception.py, robot_sim.py
│   ├── teb_controller.py, TebSolver.py
│   ├── rl_controller.py, rl_local_controller.py, final.zip
│   ├── maps/, teb.rviz, run_headless.py
│   └── tests/
├── training/       RL environment and PPO training (stage 2)
│   ├── nav_min_mujoco.py, rl_constants.py, rl_controller.py
│   └── models/mobile_robot.xml
├── docs/           Per-stage documentation
└── THIRD_PARTY_NOTICES.md
```

`rl_controller.py` exists in both `ros_nav/` and `training/`. The two copies differ only in where the constants come from. The constants `N_BEAMS`, `RANGE`, `ROOM`, `V_MAX` and `W_MAX` and the observation and action code must be identical in training and in ROS. If you change any of them, retrain the policy.

## Quick start

Tested on Ubuntu (WSL2 on Windows 11) with ROS 2. Use whichever ROS 2 distro you have installed.

```bash
source /opt/ros/<distro>/setup.bash
pip install mujoco numpy pyyaml pillow casadi matplotlib stable-baselines3
```

If you use a virtual environment, create it with `--system-site-packages` so that `rclpy` stays visible.

Run the TEB planner:

```bash
cd ros_nav
python3 ros_node.py --ros-args -p seed:=7
```

Run the RL planner on the same hidden obstacles (the trained model is `final.zip`):

```bash
python3 ros_node.py --ros-args -p local_planner:=rl -p rl_model:=final.zip -p seed:=7 -p rl_range_offset:=0.08
```

In a second terminal:

```bash
source /opt/ros/<distro>/setup.bash
rviz2 -d teb.rviz
```

Click **2D Goal Pose** on a free spot of the map. To train your own policy, see [Training the policy](docs/iteration3-rl.md#training-the-policy).

## Offline checks

These need no ROS, and the first two need no MuJoCo either. Run them from `ros_nav/`.

```bash
python3 tests/check_global_planner.py
python3 tests/check_closed_loop.py 1 new
python3 tests/check_rl_plumbing.py 4
```

`check_rl_plumbing.py` tests the wiring around the policy (lidar to 24 beams, observation size, closed loop with stand-in policies). It does not test the quality of the trained model. Use `python nav_min_mujoco.py eval` in `training/` for that.

## Known limitations

- The policy is reactive and has no memory. It was trained in an empty square room with a few boxes and has never seen corridors or doorways, so the global path carries most of the navigation.
- The training robot (cylinder, radius 0.16 m) is smaller than the ROS robot (0.40 x 0.28 m box). `rl_range_offset` compensates, and training on the larger robot would be the proper fix.
- The RL policy cannot reverse and is limited to 0.26 m/s. TEB cannot reverse or make U-turns either, which is why it rotates on the spot first.
- Obstacles are assumed to be static once seen.
- Tested in simulation only. It has not been run on a real robot.

## References and acknowledgements

### Code this repository is based on

- **Minimal_TEB** by whsleep. `ros_nav/TebSolver.py` is based on this CasADi and IPOPT implementation of the TEB optimiser, with local changes (empty obstacle lists handled, IPOPT output silenced, solver statistics kept). Used under the MIT licence, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). https://github.com/whsleep/Minimal_TEB
- **Stable-Baselines3** provides the PPO implementation used for training. https://github.com/DLR-RM/stable-baselines3
- **AutoNavRL** is a reinforcement learning navigation project listed among the projects built on the IR-SIM simulator. It was used as a reference for how to set up RL-based navigation. https://github.com/harshmahesheka/AutoNavRL
- **IR-SIM** (Intelligent Robot Simulator), the lightweight 2D simulator that AutoNavRL builds on. https://github.com/hanruihua/ir-sim
- **MuJoCo**, **Gymnasium**, **CasADi**, **IPOPT** and **ROS 2** are used as libraries (see below).

### Papers and tools

**TEB**

- C. Rösmann, W. Feiten, T. Wösch, F. Hoffmann, T. Bertram. *Trajectory modification considering dynamic constraints of autonomous robots.* ROBOTIK 2012.
- C. Rösmann, W. Feiten, T. Wösch, F. Hoffmann, T. Bertram. *Efficient trajectory optimization using a sparse model.* European Conference on Mobile Robots (ECMR), 2013.
- C. Rösmann, F. Hoffmann, T. Bertram. *Kinodynamic trajectory optimization and control for car-like robots.* IEEE/RSJ IROS, 2017.

**Reinforcement learning**

- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
- A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann. *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* Journal of Machine Learning Research 22(268), 2021.
- M. Towers et al. *Gymnasium.* https://github.com/Farama-Foundation/Gymnasium

**Simulation, planning and optimisation**

- E. Todorov, T. Erez, Y. Tassa. *MuJoCo: A physics engine for model-based control.* IEEE/RSJ IROS, 2012.
- P. E. Hart, N. J. Nilsson, B. Raphael. *A formal basis for the heuristic determination of minimum cost paths.* IEEE Transactions on Systems Science and Cybernetics, 1968. (A* global planner)
- J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, M. Diehl. *CasADi: a software framework for nonlinear optimization and optimal control.* Mathematical Programming Computation 11, 2019.
- A. Wächter, L. T. Biegler. *On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming.* Mathematical Programming 106, 2006. (IPOPT)
- S. Macenski, T. Foote, B. Gerkey, C. Lalancette, W. Woodall. *Robot Operating System 2: Design, architecture, and uses in the wild.* Science Robotics 7(66), 2022.
- H. Han et al. *IR-SIM: A Lightweight Skill-Native Simulator for Navigation, Learning, and Benchmarking.* arXiv:2606.08729.

## Licence

No licence has been chosen for this repository's own code. Third-party code is used under its own licence, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

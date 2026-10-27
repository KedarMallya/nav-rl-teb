"""Constants shared by training (nav_min_mujoco.py) and the ROS side (rl_controller.py).

The policy only works if both sides agree on these, so they live in exactly one place.
"""
N_BEAMS = 24      # lidar beams fed to the policy, beam 0 straight ahead, counter-clockwise
RANGE = 3.5       # max lidar range [m]; no return is reported as RANGE
ROOM = 6.0        # goal distance is normalised by this [m]
V_MAX = 0.26      # forward speed at action = +1 [m/s]
W_MAX = 1.82      # turn rate at action = +-1 [rad/s]

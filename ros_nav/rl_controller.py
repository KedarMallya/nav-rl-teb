"""Observation building, LiDAR conversion and policy wrapper for the trained PPO agent.

The constants MUST match nav_min_mujoco.py (the environment the policy was trained in).
"""
from __future__ import annotations
import numpy as np

N_BEAMS, RANGE, ROOM, V_MAX, W_MAX = 24, 3.5, 6.0, 0.26, 1.82


def scan_to_beams(ranges, angle_min, angle_inc, range_min=0.0, n=N_BEAMS, window=1):
    """Resample a ROS LaserScan (any resolution) to the n beams used in training.
    Beam i points at angle i*2pi/n counter-clockwise from straight ahead. Returns metres,
    with 'no return' / inf / out-of-range replaced by RANGE."""
    r = np.asarray(ranges, dtype=float)
    r = np.where(np.isfinite(r) & (r > max(range_min, 1e-3)), np.minimum(r, RANGE), RANGE)
    N = len(r)
    full = N * angle_inc >= 2 * np.pi - 2 * angle_inc
    out = np.full(n, RANGE)
    for i in range(n):
        idx = int(round(((2 * np.pi * i / n - angle_min) % (2 * np.pi)) / angle_inc))
        if full:
            ids = [(idx + k) % N for k in range(-window, window + 1)]
        elif idx < N:
            ids = [k for k in range(idx - window, idx + window + 1) if 0 <= k < N]
        else:
            continue
        out[i] = r[ids].min()
    return out


def to_local(pose, point):
    """World point -> robot frame, pose = (x, y, theta)."""
    x, y, th = pose
    dx, dy = point[0] - x, point[1] - y
    return np.array([np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy])


def build_obs(beams_m, goal_local, v, w):
    dist = float(np.hypot(*goal_local))
    bearing = float(np.arctan2(goal_local[1], goal_local[0]))
    return np.concatenate([
        np.clip(np.asarray(beams_m, float), 0.0, RANGE) / RANGE,
        [min(dist / ROOM, 1.0), np.sin(bearing), np.cos(bearing),
         np.clip(v / V_MAX, -1, 1), np.clip(w / W_MAX, -1, 1)],
    ]).astype(np.float32)


def action_to_cmd(a):
    a = np.clip(np.asarray(a, float), -1.0, 1.0)
    return 0.5 * (a[0] + 1.0) * V_MAX, a[1] * W_MAX  # (v [m/s], w [rad/s])


def goal_seeking_policy(obs):
    """Obstacle-blind fallback / baseline controller."""
    bearing = np.arctan2(obs[N_BEAMS + 1], obs[N_BEAMS + 2])
    return np.array([1.0 if abs(bearing) < 0.5 else -1.0,
                     float(np.clip(2.0 * bearing, -1, 1))], dtype=np.float32)


class Policy:
    """Loads a Stable-Baselines3 PPO model; falls back to goal_seeking_policy if no path."""

    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path, device="cpu")

    def __call__(self, obs):
        if self.model is None:
            return goal_seeking_policy(obs)
        return self.model.predict(obs, deterministic=True)[0]

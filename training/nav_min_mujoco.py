"""Minimal MuJoCo + Gymnasium + PPO mobile-robot navigation (training environment + trainer).

Usage:
  python nav_min_mujoco.py selftest
  python nav_min_mujoco.py baseline
  python nav_min_mujoco.py train --steps 500000
  python nav_min_mujoco.py eval --model runs/min_mujoco/final.zip
  python nav_min_mujoco.py eval --model runs/min_mujoco/final.zip --render

Actions are normalized left/right wheel angular-speed targets. Observations are
24 normalized MuJoCo rangefinder readings + goal distance/bearing + body velocities.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import mujoco

from rl_constants import N_BEAMS, RANGE, ROOM, V_MAX, W_MAX
from rl_controller import goal_seeking_policy

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

RADIUS = 0.20
WHEEL_RADIUS = 0.05
TRACK = 0.26
WHEEL_MAX = 10.0
DT_CONTROL = 0.1
MAX_OBST = 5
MAX_STEPS = 400
OBS_DIM = N_BEAMS + 5
MODEL_PATH = Path(__file__).parent / "models" / "mobile_robot.xml"


class NavEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, seed=0, lidar_noise=0.01, render_mode=None):
        if gym is None:
            raise ImportError("Install gymnasium: pip install gymnasium")
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.noise = float(lidar_noise)
        self.render_mode = render_mode
        self.viewer = None
        self.difficulty = 0.0
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), dtype=np.float32)
        self.lidar_ids = np.array([
            int(self.model.sensor(f"lidar_{i:02d}").adr[0]) for i in range(N_BEAMS)
        ], dtype=int)
        self.obstacle_ids = [self.model.geom(f"obstacle_{i}").id for i in range(MAX_OBST)]
        self.mocap_ids = [int(self.model.body_mocapid[self.model.body(f"obstacle_{i}_body").id])
                          for i in range(MAX_OBST)]
        self.base_body_id = self.model.body("base").id
        self.goal_site_id = self.model.site("goal_marker").id
        self.left_joint_id = self.model.joint("left_wheel_hinge").id
        self.right_joint_id = self.model.joint("right_wheel_hinge").id
        self.left_act_id = self.model.actuator("left_wheel_velocity").id
        self.right_act_id = self.model.actuator("right_wheel_velocity").id
        self.base_qpos_adr = int(self.model.joint("base_free").qposadr[0])
        self.base_dof_adr = int(self.model.joint("base_free").dofadr[0])
        self.left_qvel_adr = int(self.model.joint("left_wheel_hinge").dofadr[0])
        self.right_qvel_adr = int(self.model.joint("right_wheel_hinge").dofadr[0])
        # ids used by _collision(), looked up once instead of on every step
        self.wall_ids = {self.model.geom(n).id for n in ("wall_n", "wall_s", "wall_e", "wall_w")}
        self.robot_ids = {self.model.geom("chassis").id, self.model.geom("left_wheel_geom").id,
                          self.model.geom("right_wheel_geom").id}
        self._set_obstacle_count(0)

    def set_difficulty(self, level):
        self.difficulty = float(np.clip(level, 0, 1))

    def _set_obstacle_count(self, n):
        # Obstacles are mocap bodies (moved through data.mocap_pos). Active slots are
        # enabled; unused slots are parked below the floor with collisions disabled.
        for i in range(MAX_OBST):
            self._park(i) if i >= n else self._enable(i)

    def _enable(self, i):
        gid = self.obstacle_ids[i]
        self.data.mocap_pos[self.mocap_ids[i]] = [0.0, 0.0, 0.15]
        self.model.geom_contype[gid] = 1
        self.model.geom_conaffinity[gid] = 1

    def _park(self, i):
        gid = self.obstacle_ids[i]
        self.data.mocap_pos[self.mocap_ids[i]] = [0.0, 0.0, -5.0 - i]
        self.model.geom_contype[gid] = 0
        self.model.geom_conaffinity[gid] = 0

    def _base_xytheta(self):
        p = self.data.xpos[self.base_body_id]
        # Body matrix's first column is the local x axis.
        mat = self.data.xmat[self.base_body_id].reshape(3, 3)
        theta = float(np.arctan2(mat[1, 0], mat[0, 0]))
        return float(p[0]), float(p[1]), theta

    def _goal_local(self):
        x, y, theta = self._base_xytheta()
        dx, dy = self.goal - np.array([x, y])
        gx = np.cos(theta) * dx + np.sin(theta) * dy
        gy = -np.sin(theta) * dx + np.cos(theta) * dy
        return float(np.hypot(gx, gy)), float(np.arctan2(gy, gx))

    def _obs(self):
        # mj_sensorData is updated by mj_forward / mj_step.
        scan = self.data.sensordata[self.lidar_ids].copy()
        scan = np.nan_to_num(scan, nan=RANGE, posinf=RANGE, neginf=RANGE)
        scan = np.where(scan < 0.0, RANGE, scan)   # MuJoCo rangefinders report -1 for "no hit"
        scan = np.clip(scan, 0.0, RANGE) / RANGE
        if self.noise > 0:
            scan = np.clip(scan + self.rng.normal(0, self.noise / RANGE, N_BEAMS), 0, 1)
        dist, bearing = self._goal_local()
        # Free-joint qvel is linear xyz (world frame) followed by angular xyz (body frame).
        theta = self._base_xytheta()[2]
        vx_world = float(self.data.qvel[self.base_dof_adr + 0])
        vy_world = float(self.data.qvel[self.base_dof_adr + 1])
        v_forward = vx_world * np.cos(theta) + vy_world * np.sin(theta)
        yaw_rate = float(self.data.qvel[self.base_dof_adr + 5])
        v = float(np.clip(v_forward, -V_MAX, V_MAX) / V_MAX)
        w = float(np.clip(yaw_rate, -W_MAX, W_MAX) / W_MAX)
        return np.concatenate([scan, [min(dist / ROOM, 1.0), np.sin(bearing),
                                      np.cos(bearing), v, w]]).astype(np.float32)

    def _collision(self):
        # A contact counts if a robot geom touches a room wall or an obstacle.
        # Parked obstacle slots have collisions disabled, so they never produce contacts.
        obstacle_ids = set(self.obstacle_ids)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if ((g1 in self.robot_ids and (g2 in self.wall_ids or g2 in obstacle_ids)) or
                (g2 in self.robot_ids and (g1 in self.wall_ids or g1 in obstacle_ids))):
                return True
        x, y, _ = self._base_xytheta()
        return abs(x) > 2.8 or abs(y) > 2.8

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        n_obst = int(round(self.difficulty * MAX_OBST))
        self._set_obstacle_count(n_obst)
        # Randomize active obstacle positions, rejecting placements near start/goal.
        self.start = np.array([self.rng.uniform(-2.3, 2.3), self.rng.uniform(-2.3, 2.3)])
        self.goal = np.array([self.rng.uniform(-2.3, 2.3), self.rng.uniform(-2.3, 2.3)])
        for _ in range(100):
            if np.linalg.norm(self.goal - self.start) > 1.5:
                break
            self.goal = self.rng.uniform(-2.3, 2.3, size=2)
        placed = []
        for i in range(n_obst):
            for _ in range(100):
                p = self.rng.uniform(-2.2, 2.2, size=2)
                if (np.linalg.norm(p - self.start) > 0.75 and
                        np.linalg.norm(p - self.goal) > 0.55 and
                        all(np.linalg.norm(p - q) > 0.65 for q in placed)):
                    self.data.mocap_pos[self.mocap_ids[i]] = [p[0], p[1], 0.15]
                    placed.append(p)
                    break
            else:
                self._park(i)  # could not place without overlap: leave this slot unused
        self.model.site_pos[self.goal_site_id] = [self.goal[0], self.goal[1], 0.01]  # visual only
        qadr = self.base_qpos_adr
        self.data.qpos[qadr:qadr+3] = [self.start[0], self.start[1], 0.12]
        # Set freejoint yaw quaternion in MuJoCo order [w, x, y, z].
        yaw = self.rng.uniform(-np.pi, np.pi)
        self.data.qpos[qadr+3:qadr+7] = [np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        self.d_prev = self._goal_local()[0]
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # Differential drive: convert normalized forward/turn action to wheel speeds.
        v_cmd = 0.5 * (float(a[0]) + 1.0) * V_MAX
        w_cmd = float(a[1]) * W_MAX
        wl = np.clip((v_cmd - 0.5 * TRACK * w_cmd) / WHEEL_RADIUS, -WHEEL_MAX, WHEEL_MAX)
        wr = np.clip((v_cmd + 0.5 * TRACK * w_cmd) / WHEEL_RADIUS, -WHEEL_MAX, WHEEL_MAX)
        self.data.ctrl[self.left_act_id] = wl
        self.data.ctrl[self.right_act_id] = wr
        n_substeps = max(1, int(round(DT_CONTROL / self.model.opt.timestep)))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        d = self._goal_local()[0]
        reward = 5.0 * (self.d_prev - d) - 0.01
        self.d_prev = d
        collided = self._collision()
        reached = d < 0.25
        if collided:
            reward -= 10.0
        if reached:
            reward += 10.0
        terminated = bool(collided or reached)
        truncated = bool((not terminated) and self.t >= MAX_STEPS)
        info = {"success": bool(reached), "collision": bool(collided)}
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.viewer is None:
            try:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                raise RuntimeError("MuJoCo viewer unavailable; run eval on a desktop with GUI.") from e
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def cmd_selftest(_):
    env = NavEnv(seed=0, lidar_noise=0)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,), obs.shape
        assert np.isfinite(obs).all(), obs
        for _ in range(5):
            obs, _, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
            assert np.isfinite(obs).all()
            if term or trunc:
                break
        from gymnasium.utils.env_checker import check_env
        check_env(NavEnv(seed=1), skip_render_check=True)
        print("MuJoCo environment selftest passed; observation shape:", obs.shape)
    finally:
        env.close()


def run_episodes(env, policy, n_obstacles, episodes, seed):
    env.set_difficulty(n_obstacles / MAX_OBST)
    results = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        results.append("success" if info["success"] else
                       "collision" if info["collision"] else "timeout")
    return {k: results.count(k) / len(results) for k in ("success", "collision", "timeout")}


def cmd_baseline(a):
    env = NavEnv(seed=a.seed, lidar_noise=0)
    try:
        for n in (0, 3, 5):
            print(f"Goal-seeking baseline, {n} obstacles:",
                  run_episodes(env, goal_seeking_policy, n, a.episodes, a.seed))
    finally:
        env.close()


def make_vec_env(n_envs, seed):
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv([lambda i=i: NavEnv(seed=seed+i) for i in range(n_envs)])


def cmd_train(a):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    class Curriculum(BaseCallback):
        """Raise the obstacle count from 0 to MAX_OBST over the first half of training."""
        def _on_step(self):
            if self.n_calls % 1000 == 0:
                level = min(1.0, self.num_timesteps / max(1, 0.5 * a.steps))
                self.training_env.env_method("set_difficulty", level)
                self.logger.record("curriculum/difficulty", level)
            return True

    class Stats(BaseCallback):
        """Log success / collision rate over the last 200 finished episodes."""
        def __init__(self):
            super().__init__()
            self.buf = []
        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done and "success" in info:
                    self.buf.append((info["success"], info["collision"]))
            if len(self.buf) >= 50:
                b = np.asarray(self.buf[-200:], dtype=float)
                self.logger.record("rollout/success_rate", b[:, 0].mean())
                self.logger.record("rollout/collision_rate", b[:, 1].mean())
            return True

    env = make_vec_env(a.n_envs, a.seed)
    try:
        model = (PPO.load(a.resume, env=env, device=a.device) if a.resume else
                 PPO("MlpPolicy", env, n_steps=512, batch_size=256, n_epochs=8,
                     learning_rate=3e-4, gamma=0.99, ent_coef=0.005,
                     policy_kwargs=dict(net_arch=[64, 64]), device=a.device,
                     seed=a.seed, verbose=1, tensorboard_log=str(out / "tb")))
        ckpt = CheckpointCallback(save_freq=max(1, 50_000 // a.n_envs),
                                  save_path=str(out / "ckpt"), name_prefix="ppo")
        model.learn(total_timesteps=a.steps, callback=[Curriculum(), Stats(), ckpt],
                    reset_num_timesteps=not bool(a.resume))
        model.save(out / "final")
        print("Saved", out / "final.zip")
    finally:
        env.close()


def cmd_eval(a):
    from stable_baselines3 import PPO
    model = PPO.load(a.model, device="cpu")
    env = NavEnv(seed=a.seed, render_mode="human" if a.render else None, lidar_noise=0)
    try:
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
        for n in (0, 3, 5):
            print(f"PPO, {n} obstacles:", run_episodes(env, policy, n, a.episodes, a.seed))
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").set_defaults(func=cmd_selftest)
    b = sub.add_parser("baseline"); b.set_defaults(func=cmd_baseline)
    b.add_argument("--episodes", type=int, default=50)
    b.add_argument("--seed", type=int, default=123)
    t = sub.add_parser("train"); t.set_defaults(func=cmd_train)
    t.add_argument("--steps", type=int, default=500_000)
    t.add_argument("--n-envs", type=int, default=4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.add_argument("--resume", default=None)
    t.add_argument("--out", default="runs/min_mujoco")
    e = sub.add_parser("eval"); e.set_defaults(func=cmd_eval)
    e.add_argument("--model", required=True)
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--seed", type=int, default=123)
    e.add_argument("--render", action="store_true")
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

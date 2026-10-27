"""MuJoCo differential-drive robot with static obstacles.

World frame == RViz "map" frame (x forward/east, y left, z up).

Two obstacle modes:
  * boxes=None  : obstacles are ellipsoids built from the (m,7) solver rows
                  [x, y, a, b, theta, cos(theta), sin(theta)]  (make_obstacles()).
  * boxes=(k,4) : obstacles are boxes [cx, cy, hx, hy] (exact map cells from map_utils);
                  `obstacles` is then only kept for the planner.
"""
import numpy as np
import mujoco

WHEEL_R = 0.05      # wheel radius [m]
TRACK = 0.32        # distance between the two drive wheels [m]
ROBOT_RADIUS = 0.25  # rough footprint radius, handy for planning / checks


def make_obstacles(specs):
    """specs: iterable of (x, y, a, b, theta). Returns (m, 7) array for the solver."""
    rows = []
    for x, y, a, b, th in specs:
        rows.append([x, y, a, b, th, np.cos(th), np.sin(th)])
    return np.array(rows, dtype=float).reshape(-1, 7)


def make_mjcf(start_pose, obstacles, arena=8.0, boxes=None, floor_center=(0.0, 0.0), extras=None):
    x0, y0, yaw0 = start_pose
    fx, fy = floor_center
    obs_xml = []
    if boxes is not None:
        for k, (bx, by, hx, hy) in enumerate(boxes):
            obs_xml.append(
                f'<geom name="obs{k}" type="box" pos="{bx} {by} 0.25" size="{hx} {hy} 0.25" '
                f'rgba="0.85 0.25 0.2 1"/>'
            )
    else:
        for k, row in enumerate(obstacles):
            ox, oy, a, b, th = row[:5]
            # half-height 0.5 centred on the floor -> cross-section at robot height ~= (a, b)
            obs_xml.append(
                f'<geom name="obs{k}" type="ellipsoid" pos="{ox} {oy} 0" size="{a} {b} 0.5" '
                f'euler="0 0 {th}" rgba="0.85 0.25 0.2 1"/>'
            )
    # extra (random, hidden-from-the-map) obstacles: rows [kind, x, y, sx, sy, yaw]
    # kind 0 = cylinder (sx = radius), 1 = box (half sizes sx, sy). Names start with "obs" so
    # obstacle_contacts() counts them.
    for k, (kind, ex, ey, sx, sy, yaw) in enumerate([] if extras is None else extras):
        if int(kind) == 0:
            obs_xml.append(f'<geom name="obs_rnd{k}" type="cylinder" pos="{ex} {ey} 0.25" '
                           f'size="{sx} 0.25" rgba="0.8 0.2 0.85 1"/>')
        else:
            obs_xml.append(f'<geom name="obs_rnd{k}" type="box" pos="{ex} {ey} 0.25" '
                           f'size="{sx} {sy} 0.25" euler="0 0 {yaw}" rgba="0.8 0.2 0.85 1"/>')
    obs_xml = "\n    ".join(obs_xml)
    half = TRACK / 2

    return f"""
<mujoco model="teb_diffdrive">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <statistic center="{fx} {fy} 0" extent="{2 * arena}"/>
  <visual>
    <global elevation="-75"/>
  </visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1=".22 .30 .38" rgb2=".27 .36 .45"
             width="300" height="300" mark="edge" markrgb=".8 .8 .8"/>
    <material name="grid" texture="grid" texrepeat="{int(arena) + 1} {int(arena) + 1}" reflectance=".05"/>
  </asset>
  <worldbody>
    <light pos="{fx} {fy} 8" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" pos="{fx} {fy} 0" size="{arena} {arena} 0.1" material="grid"
          friction="1 0.005 0.0001"/>
    {obs_xml}
    <body name="base" pos="{x0} {y0} {WHEEL_R}" euler="0 0 {yaw0}">
      <freejoint name="root"/>
      <geom name="chassis" type="box" size="0.2 0.14 0.04" pos="0 0 0.04" mass="4"
            rgba="0.2 0.5 0.9 1"/>
      <geom name="nose" type="box" size="0.04 0.10 0.02" pos="0.17 0 0.09" mass="0.01"
            rgba="1 0.8 0.1 1" contype="0" conaffinity="0"/>
      <geom name="caster_f" type="sphere" size="0.02" pos="0.17 0 -0.028" mass="0.1"
            friction="0.02 0.001 0.0001" rgba="0.3 0.3 0.3 1"/>
      <geom name="caster_r" type="sphere" size="0.02" pos="-0.17 0 -0.028" mass="0.1"
            friction="0.02 0.001 0.0001" rgba="0.3 0.3 0.3 1"/>
      <body name="wheel_l" pos="0 {half} 0">
        <joint name="wl" type="hinge" axis="0 1 0" damping="0.001" armature="0.005"/>
        <geom name="wheel_l_geom" type="cylinder" size="{WHEEL_R} 0.015" euler="1.5708 0 0"
              mass="0.3" friction="1.2 0.005 0.0001" rgba="0.1 0.1 0.1 1"/>
        <geom type="box" size="0.02 0.017 0.006" pos="{WHEEL_R*0.7} 0 0" mass="0.001"
              rgba="1 1 1 1" contype="0" conaffinity="0"/>
      </body>
      <body name="wheel_r" pos="0 {-half} 0">
        <joint name="wr" type="hinge" axis="0 1 0" damping="0.001" armature="0.005"/>
        <geom name="wheel_r_geom" type="cylinder" size="{WHEEL_R} 0.015" euler="1.5708 0 0"
              mass="0.3" friction="1.2 0.005 0.0001" rgba="0.1 0.1 0.1 1"/>
        <geom type="box" size="0.02 0.017 0.006" pos="{WHEEL_R*0.7} 0 0" mass="0.001"
              rgba="1 1 1 1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="wl_act" joint="wl" kv="1.0" ctrlrange="-40 40" forcerange="-3 3"/>
    <velocity name="wr_act" joint="wr" kv="1.0" ctrlrange="-40 40" forcerange="-3 3"/>
  </actuator>
</mujoco>
"""


class DiffDriveSim:
    """Thin wrapper: set (v, w), step, read pose. Positive wheel speed = forward."""

    def __init__(self, start_pose=(0.0, 0.0, 0.0), obstacles=None, boxes=None,
                 floor_center=(0.0, 0.0), arena=8.0, extras=None):
        self.obstacles = make_obstacles([]) if obstacles is None else np.asarray(obstacles)
        self.model = mujoco.MjModel.from_xml_string(
            make_mjcf(start_pose, self.obstacles, arena=arena, boxes=boxes,
                      floor_center=floor_center, extras=extras))
        self.data = mujoco.MjData(self.model)
        self._base_id = self.model.body("base").id
        self._act_l = self.model.actuator("wl_act").id
        self._act_r = self.model.actuator("wr_act").id
        mujoco.mj_forward(self.model, self.data)

    # --- command ---------------------------------------------------------
    def set_cmd(self, v, w):
        """v [m/s] forward, w [rad/s] counter-clockwise."""
        self.data.ctrl[self._act_l] = (v - w * TRACK / 2) / WHEEL_R
        self.data.ctrl[self._act_r] = (v + w * TRACK / 2) / WHEEL_R

    def step(self, duration):
        n = max(1, int(round(duration / self.model.opt.timestep)))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # --- state -----------------------------------------------------------
    @property
    def time(self):
        return float(self.data.time)

    @property
    def pose(self):
        """[x, y, yaw] of the base in the world/map frame."""
        x, y, _ = self.data.xpos[self._base_id]
        w, qx, qy, qz = self.data.xquat[self._base_id]
        yaw = np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return np.array([x, y, yaw])

    @property
    def velocity(self):
        """(v, w) in the body frame, taken from the free joint's qvel."""
        yaw = self.pose[2]
        vx, vy = self.data.qvel[0], self.data.qvel[1]
        return float(vx * np.cos(yaw) + vy * np.sin(yaw)), float(self.data.qvel[5])

    def obstacle_contacts(self):
        """Number of active contacts between the robot and any obstacle geom."""
        count = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if n1.startswith("obs") or n2.startswith("obs"):
                count += 1
        return count

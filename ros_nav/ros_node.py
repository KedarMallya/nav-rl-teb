#!/usr/bin/env python3
"""ROS 2 node: RViz 2D Goal Pose -> TEB planner -> MuJoCo diff-drive robot, on a loaded map.

Subscribes : /goal_pose            (geometry_msgs/PoseStamped, RViz "2D Goal Pose" tool)
Publishes  : /map                  (nav_msgs/OccupancyGrid, latched)  <- the loaded map
             /odom                 (nav_msgs/Odometry, frame `map`)
             TF map -> base_link
             /teb_path             (nav_msgs/Path, the current optimised trajectory)
             /global_path          (nav_msgs/Path, A* path on the map + sensed obstacles)
             /teb_obstacles        (visualization_msgs/MarkerArray, latched) what the PLANNER sees
             /teb_goal             (visualization_msgs/Marker)
             /cmd_vel              (geometry_msgs/Twist, for inspection)
             /scan                 (sensor_msgs/LaserScan, simulated lidar, frame base_link)
             /teb_truth            (MarkerArray) random obstacles that really exist in MuJoCo (magenta)
             /teb_sensed           (MarkerArray) obstacles the lidar has discovered (cyan) = what the
                                   planner avoids in addition to the map

Run:  python3 ros_node.py
Parameters (--ros-args -p name:=value):
  map_yaml:=/path/to/map.yaml   map_server-style YAML (+ PGM/PNG). Default: maps/demo.yaml (auto-created)
  rect_max_len:=1.0             max side [m] of the rectangles the map is split into
  n_random:=8  seed:=-1         number of random hidden obstacles; seed -1 = new layout each run
  show_truth:=false             debug only: draw the true random obstacles in RViz (magenta).
                                Default off: RViz gets the static map, the robot must sense the rest
  lidar_range:=4.0  scan_rate:=10.0
  local_planner:=teb|rl  rl_model:=/path/final.zip  rl_lookahead:=1.5  rl_range_offset:=0.0
                                RL = PPO policy (rl_controller.py) instead of TEB; needs stable-baselines3
  sensed_pad:=0.08 sensed_shift:=0.15 sensed_clearance:=0.30   room left around random obstacles
                                [m]; raise them if the robot clips obstacles, lower to squeeze closer
  start_x:=0.0 start_y:=0.0 start_yaw:=0.0   (use decimals)
  viewer:=false  goal_topic:=/goal_pose  frame:=map
In RViz set Fixed Frame = map and add a Map display on /map (Durability: Transient Local).
"""
import contextlib
import math
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from map_utils import (distance_to_rects, grid_to_rects, load_map,
                       rects_to_ellipses, write_demo_map)
from global_planner import GlobalPlanner
from perception import Lidar, ObstacleTracker, extras_to_circles, random_obstacles
from robot_sim import DiffDriveSim, ROBOT_RADIUS
from teb_controller import TebController

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MAP = os.path.join(HERE, "maps", "demo.yaml")


def yaw_to_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)  # x, y, z, w


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TebNode(Node):
    def __init__(self):
        super().__init__("teb_mujoco")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("viewer", True)
        self.declare_parameter("plan_rate", 10.0)
        self.declare_parameter("sim_rate", 50.0)
        self.declare_parameter("map_yaml", "")
        self.declare_parameter("rect_max_len", 1.0)
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("start_yaw", 0.0)
        self.declare_parameter("n_random", 8)
        self.declare_parameter("seed", -1)
        self.declare_parameter("show_truth", False)
        self.declare_parameter("lidar_range", 4.0)
        self.declare_parameter("scan_rate", 10.0)
        # how much room the planners leave around SENSED (random) obstacles
        self.declare_parameter("sensed_pad", 0.08)        # tracker: extra radius on each circle [m]
        self.declare_parameter("sensed_shift", 0.15)      # tracker: centre pushed behind the seen face [m]
        self.declare_parameter("local_planner", "teb")    # "teb" or "rl"
        self.declare_parameter("rl_model", "")            # PPO .zip; empty -> obstacle-blind fallback
        self.declare_parameter("rl_lookahead", 1.5)       # carrot distance along the global path [m]
        self.declare_parameter("rl_range_offset", 0.0)    # make obstacles look this much closer [m]
        self.declare_parameter("sensed_clearance", 0.30)  # TEB: wanted distance robot-centre <-> circle edge [m]

        self.frame = self.get_parameter("frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.use_viewer = self.get_parameter("viewer").value
        self.plan_period = 1.0 / self.get_parameter("plan_rate").value
        self.sim_dt = 1.0 / self.get_parameter("sim_rate").value

        # ---- map -> rectangles -> (MuJoCo boxes, planner ellipses) ----------------
        map_yaml = self.get_parameter("map_yaml").value or DEFAULT_MAP
        if not os.path.exists(map_yaml):
            if map_yaml != DEFAULT_MAP:
                raise FileNotFoundError(map_yaml)
            write_demo_map(map_yaml)
            self.get_logger().info(f"Created demo map {map_yaml}")
        self.gridmap = load_map(map_yaml)
        self.rects = grid_to_rects(self.gridmap, float(self.get_parameter("rect_max_len").value))
        self.obstacles = rects_to_ellipses(self.rects)
        ex, ey = self.gridmap.extent
        self.get_logger().info(
            f"Map {map_yaml}: {self.gridmap.width}x{self.gridmap.height} cells "
            f"({ex:.1f}x{ey:.1f} m, res {self.gridmap.resolution}), "
            f"{len(self.rects)} obstacle rectangles")

        start = (float(self.get_parameter("start_x").value),
                 float(self.get_parameter("start_y").value),
                 float(self.get_parameter("start_yaw").value))
        if distance_to_rects(start[:2], self.rects) < ROBOT_RADIUS:
            self.get_logger().warn(f"Start pose {start} is inside/too close to an obstacle in the map!")

        # ---- random obstacles: exist in MuJoCo only; the planner must sense them ------
        seed = int(self.get_parameter("seed").value)
        if seed < 0:
            seed = int(time.time()) % 100000
        ox, oy = self.gridmap.origin[:2]
        bounds = (ox + 0.6, oy + 0.6, ox + ex - 0.6, oy + ey - 0.6)
        self.extras = random_obstacles(self.rects, bounds, int(self.get_parameter("n_random").value),
                                       seed, keep_out=[(start[0], start[1], ROBOT_RADIUS)])
        self.truth_circles = extras_to_circles(self.extras)
        self.get_logger().info(f"Random obstacles: {len(self.extras)} (seed {seed}; "
                               f"reuse with -p seed:={seed})")

        self.sim = DiffDriveSim(start, self.obstacles, boxes=self.rects,
                                floor_center=self.gridmap.center,
                                arena=0.5 * max(ex, ey) + 0.5, extras=self.extras)
        self.lidar = Lidar(self.sim.model, self.sim.data,
                           max_range=float(self.get_parameter("lidar_range").value))
        self.tracker = ObstacleTracker(self.gridmap,
                                       pad=float(self.get_parameter("sensed_pad").value),
                                       shift=float(self.get_parameter("sensed_shift").value))
        self.scan_every = max(1, int(round(self.get_parameter("sim_rate").value
                                           / float(self.get_parameter("scan_rate").value))))
        self._tick = 0
        self._last_sensed = 0
        # global A* on the map (+ lidar-discovered obstacles) feeds TEB its local goals
        self.global_planner = GlobalPlanner(self.gridmap)
        if self.get_parameter("local_planner").value == "rl":
            from rl_controller import Policy
            from rl_local_controller import RLController
            policy = Policy(self.get_parameter("rl_model").value or None)
            if policy.model is None:
                self.get_logger().warn("local_planner=rl but no rl_model given: using the "
                                       "obstacle-blind goal-seeking fallback!")
            self.controller = RLController(
                self.global_planner, policy,
                lookahead=float(self.get_parameter("rl_lookahead").value),
                range_offset=float(self.get_parameter("rl_range_offset").value),
                control_dt=self.plan_period)
            self.sensed_offset = 0.0
        else:
            self.controller = TebController(self.obstacles, global_planner=self.global_planner)
            # TEB adds its own safe_distance (default 0.5 m) to every obstacle radius. For sensed
            # circles we shift the radius so the soft wall sits at `sensed_clearance` instead.
            self.sensed_offset = (float(self.get_parameter("sensed_clearance").value)
                                  - self.controller.solver_kwargs["safe_distance"])
        self._last_gpath_version = -1
        self.grid_msg = self.build_grid_msg()

        self.lock = threading.Lock()
        self.pose = self.sim.pose
        self.vel = (0.0, 0.0)       # measured (v, w), for the RL observation
        self.scan = None            # latest (ranges, angle_min, angle_inc, range_min)
        self.cmd = (0.0, 0.0)
        self.running = True

        self.tf_br = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.path_pub = self.create_publisher(Path, "/teb_path", 1)
        self.gpath_pub = self.create_publisher(Path, "/global_path", 1)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.goal_marker_pub = self.create_publisher(Marker, "/teb_goal", 1)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.obs_pub = self.create_publisher(MarkerArray, "/teb_obstacles", latched)
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", latched)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 5)
        self.truth_pub = self.create_publisher(MarkerArray, "/teb_truth", latched)
        self.sensed_pub = self.create_publisher(MarkerArray, "/teb_sensed", 1)
        self.create_subscription(PoseStamped, self.get_parameter("goal_topic").value,
                                 self.on_goal, 10)
        self.publish_static()
        self.create_timer(2.0, self.publish_static)
        self.create_timer(0.5, self.publish_sensed)
        self.get_logger().info(
            f"Ready. Use RViz '2D Goal Pose' (topic {self.get_parameter('goal_topic').value}, "
            f"Fixed Frame '{self.frame}').")

    # ---- ROS callbacks ------------------------------------------------------
    def on_goal(self, msg: PoseStamped):
        if msg.header.frame_id not in ("", self.frame):
            self.get_logger().warn(
                f"Goal is in frame '{msg.header.frame_id}', expected '{self.frame}'. "
                f"Set RViz Fixed Frame to '{self.frame}'. Ignored.")
            return
        p = msg.pose.position
        goal = np.array([p.x, p.y, quat_to_yaw(msg.pose.orientation)])
        if distance_to_rects(goal[:2], self.rects) < ROBOT_RADIUS + 0.05:
            self.get_logger().warn(
                f"Goal ({goal[0]:.2f}, {goal[1]:.2f}) is inside or too close to an obstacle. Ignored.")
            return
        if len(self.truth_circles) and np.min(
                np.hypot(self.truth_circles[:, 0] - goal[0], self.truth_circles[:, 1] - goal[1])
                - self.truth_circles[:, 2]) < ROBOT_RADIUS + 0.05:
            self.get_logger().warn("Goal is inside a random obstacle. Ignored.")
            return
        self.controller.set_goal(goal)
        self.get_logger().info(f"New goal: x={goal[0]:.2f} y={goal[1]:.2f} yaw={goal[2]:.2f}")
        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "goal", 0, Marker.ARROW, Marker.ADD
        m.pose = msg.pose
        m.scale.x, m.scale.y, m.scale.z = 0.5, 0.08, 0.08
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 1.0
        self.goal_marker_pub.publish(m)

    # ---- simulation tick (called from the main thread) ------------------------
    def sim_tick(self):
        with self.lock:
            v, w = self.cmd
        self.sim.set_cmd(v, w)
        self.sim.step(self.sim_dt)
        pose = self.sim.pose
        vel = self.sim.velocity
        with self.lock:
            self.pose, self.vel = pose, vel
        self.publish_state(pose, v, w)
        self._tick += 1
        if self._tick % self.scan_every == 0:  # runs in the thread that steps MuJoCo
            ranges = self.lidar.scan(pose)
            with self.lock:
                self.scan = (ranges, float(self.lidar.angles[0]),
                             float(2 * math.pi / self.lidar.n), 0.05)
            self.tracker.update(pose, ranges, self.lidar.angles)
            self.publish_scan(ranges)

    def publish_state(self, pose, v_cmd, w_cmd):
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = yaw_to_quat(pose[2])
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, self.frame, self.base_frame
        tf.transform.translation.x, tf.transform.translation.y = float(pose[0]), float(pose[1])
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.tf_br.sendTransform(tf)

        v, w = self.sim.velocity
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, self.frame, self.base_frame
        od.pose.pose.position.x, od.pose.pose.position.y = float(pose[0]), float(pose[1])
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = qz, qw
        od.twist.twist.linear.x, od.twist.twist.angular.z = v, w
        self.odom_pub.publish(od)

        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v_cmd), float(w_cmd)
        self.cmd_pub.publish(cmd)

    # ---- static things: the map and the planner's view of it -------------------
    def build_grid_msg(self):
        m = self.gridmap
        g = OccupancyGrid()
        g.header.frame_id = self.frame
        g.info.resolution = float(m.resolution)
        g.info.width, g.info.height = int(m.width), int(m.height)
        g.info.origin.position.x, g.info.origin.position.y = float(m.origin[0]), float(m.origin[1])
        g.info.origin.orientation.w = 1.0
        g.data = (m.occ.astype(np.int8) * 100).ravel().tolist()  # row-major, row 0 = bottom
        return g

    def publish_static(self):
        stamp = self.get_clock().now().to_msg()
        self.grid_msg.header.stamp = stamp
        self.grid_msg.info.map_load_time = stamp
        self.map_pub.publish(self.grid_msg)
        self.publish_obstacles(self.obstacles)
        self.publish_truth()

    def publish_scan(self, ranges):
        s = LaserScan()
        s.header.stamp = self.get_clock().now().to_msg()
        s.header.frame_id = self.base_frame
        s.angle_min = float(self.lidar.angles[0])
        s.angle_increment = float(2 * math.pi / self.lidar.n)
        s.angle_max = s.angle_min + s.angle_increment * (self.lidar.n - 1)
        s.scan_time = 1.0 / float(self.get_parameter("scan_rate").value)
        s.range_min, s.range_max = 0.05, float(self.lidar.max_range)
        s.ranges = [float(r) for r in ranges]
        self.scan_pub.publish(s)

    def _circle_markers(self, circles, ns, rgba):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for k, (x, y, r) in enumerate(circles):
            m = Marker()
            m.header.frame_id = self.frame
            m.ns, m.id, m.type, m.action = ns, k, Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.25
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2 * float(r)
            m.scale.z = 0.5
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            arr.markers.append(m)
        return arr

    def publish_truth(self):
        if not self.get_parameter("show_truth").value:
            return
        self.truth_pub.publish(self._circle_markers(self.truth_circles, "truth", (0.8, 0.2, 0.85, 0.5)))

    def publish_sensed(self):
        circles = self.tracker.snapshot()
        self.sensed_pub.publish(self._circle_markers(circles, "sensed", (0.1, 0.9, 0.9, 0.45)))
        if len(circles) != self._last_sensed:
            self._last_sensed = len(circles)
            self.get_logger().info(f"lidar: {len(circles)} unknown obstacle(s) discovered "
                                   f"(of {len(self.extras)} hidden)")

    def publish_obstacles(self, obstacles):
        """Planner ellipses, drawn translucent on top of the map."""
        arr = MarkerArray()
        for k, (x, y, a, b, th, *_rest) in enumerate(obstacles):
            m = Marker()
            m.header.frame_id = self.frame
            m.ns, m.id, m.type, m.action = "obstacles", k, Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.25
            _, _, qz, qw = yaw_to_quat(th)
            m.pose.orientation.z, m.pose.orientation.w = qz, qw
            m.scale.x, m.scale.y, m.scale.z = 2 * float(a), 2 * float(b), 0.5
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.6, 0.1, 0.3
            arr.markers.append(m)
        self.obs_pub.publish(arr)

    def publish_path(self, xy, pub=None):
        path = Path()
        path.header.frame_id = self.frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y in xy:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        (pub or self.path_pub).publish(path)

    # ---- planner thread ---------------------------------------------------------
    def _planner_loop(self):
        last_status = ""
        while self.running:
            t0 = time.perf_counter()
            with self.lock:
                pose, vel, scan = self.pose.copy(), self.vel, self.scan
            sensed = self.tracker.as_solver_rows(self.sensed_offset)
            self.controller.obstacles = np.vstack([self.obstacles, sensed]) if len(sensed) else self.obstacles
            self.controller.dynamic_circles = np.array(self.tracker.snapshot(), float).reshape(-1, 3)
            v, w, path = self.controller.update(pose, self.plan_period, scan=scan, vel=vel)
            with self.lock:
                self.cmd = (v, w)
            if path is not None:
                self.publish_path(path)
            gp = self.global_planner
            if gp.path is not None and gp.version != self._last_gpath_version:
                self._last_gpath_version = gp.version
                self.publish_path(gp.path, self.gpath_pub)
            status = self.controller.last_status
            if status != last_status:
                self.get_logger().info(f"planner: {status}")
                last_status = status
            time.sleep(max(0.0, self.plan_period - (time.perf_counter() - t0)))

    def start_planner(self):
        threading.Thread(target=self._planner_loop, daemon=True).start()


def main():
    rclpy.init()
    node = TebNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    node.start_planner()

    viewer = None
    if node.use_viewer:
        import mujoco.viewer  # on macOS run this script with `mjpython`
        viewer = mujoco.viewer.launch_passive(node.sim.model, node.sim.data)

    next_t = time.perf_counter()
    try:
        while rclpy.ok() and (viewer is None or viewer.is_running()):
            with (viewer.lock() if viewer else contextlib.nullcontext()):
                node.sim_tick()
            if viewer:
                viewer.sync()
            next_t += node.sim_dt  # real-time pacing
            time.sleep(max(0.0, next_t - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

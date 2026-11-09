#!/usr/bin/env python3
"""GeoSemTwin-X ROS 2 node: RGB-D stream in, live semantic digital twin out.

Subscribes : rgb / depth images + camera_info (any RGB-D camera or the dataset player), optional initial pose
Publishes  : /geosemtwinx/pose (PoseStamped), TF map -> camera, /geosemtwinx/path (Path),
             /geosemtwinx/map (PointCloud2, fused map), /geosemtwinx/assets (MarkerArray, persistent assets),
             /geosemtwinx/traversability (OccupancyGrid), /geosemtwinx/nbv_goal (PoseStamped) + /geosemtwinx/nbv_path,
             /geosemtwinx/annotated (Image, detections), /geosemtwinx/status (String, JSON)
Services   : /geosemtwinx/save_twin (std_srvs/Trigger) -> twin store + dashboard; also done on shutdown.

Run (after `source /opt/ros/$ROS_DISTRO/setup.bash` and activating the venv):
  python3 ros2/geosemtwinx_node.py --ros-args --params-file configs/staer_player.yaml
"""
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import OccupancyGrid, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from geosemtwinx.demo import draw_camera                         # noqa: E402
from geosemtwinx.streaming import LiveTwin, mount_pose            # noqa: E402
from ros_utils import (asset_markers, image_to_numpy, make_pointcloud2, matrix_to_pose, numpy_to_image,  # noqa: E402
                       pose_to_matrix, traversability_grid)


class GeoSemTwinXNode(Node):
    def __init__(self):
        super().__init__("geosemtwinx")
        P = self.declare_parameter
        P("rgb_topic", "/camera/color/image_raw"); P("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        P("camera_info_topic", "/camera/color/camera_info"); P("initial_pose_topic", "/geosemtwinx/initial_pose")
        P("init_mode", "mount")            # mount | topic | identity
        P("camera_height", 1.0); P("camera_pitch_deg", 0.0)
        P("classes", ["forklift", "pallet", "cardboard box", "pallet truck"])
        P("floor_classes", ["forklift", "pallet truck"])
        P("detector", "grounded_sam2"); P("dino", "tiny"); P("sam", "tiny"); P("box_threshold", 0.30)
        P("detect_every", 3); P("depth_max", 6.0); P("map_voxel", 0.04); P("map_sigma_max", 0.08); P("floor_z", 0.0)
        P("map_extent", 25.0); P("tracker", "hybrid"); P("process_all", False); P("image_scale", 1.0)
        P("map_frame", "map"); P("camera_frame", "camera_optical")
        P("nbv_every_s", 5.0); P("map_every_s", 2.0); P("assets_every_s", 1.0); P("dashboard_every_s", 0.0)
        P("output_dir", "results/ros_live")
        g = lambda n: self.get_parameter(n).value
        self.g = g
        self.classes = list(g("classes"))
        self.K, self.lt, self.T_init = None, None, None
        self.queue = deque(maxlen=None if g("process_all") else 1)
        self.lock = threading.Lock()
        self.detector = None
        if g("detector") == "grounded_sam2":
            from geosemtwinx.experiments import make_detector
            self.get_logger().info("loading Grounding DINO + SAM 2.1 ...")
            self.detector = make_detector(self.classes, dino=g("dino"), sam=g("sam"), threshold=g("box_threshold"),
                                          ckpt_dir=str(ROOT / "checkpoints"))
        if g("init_mode") == "mount":
            self.T_init = mount_pose(g("camera_height"), g("camera_pitch_deg")); self.T_init[2, 3] += g("floor_z")
        elif g("init_mode") == "identity":
            self.T_init = np.eye(4)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(CameraInfo, g("camera_info_topic"), self.on_info, 10)
        self.create_subscription(PoseStamped, g("initial_pose_topic"), self.on_init_pose, latched)
        qos = 50 if g("process_all") else 5
        self.sync = ApproximateTimeSynchronizer([Subscriber(self, Image, g("rgb_topic"), qos_profile=qos),
                                                 Subscriber(self, Image, g("depth_topic"), qos_profile=qos)], qos, 0.05)
        self.sync.registerCallback(self.on_rgbd)
        self.pub_pose = self.create_publisher(PoseStamped, "/geosemtwinx/pose", 10)
        self.pub_path = self.create_publisher(PathMsg, "/geosemtwinx/path", 10)
        self.pub_map = self.create_publisher(PointCloud2, "/geosemtwinx/map", 2)
        self.pub_assets = self.create_publisher(MarkerArray, "/geosemtwinx/assets", 2)
        self.pub_grid = self.create_publisher(OccupancyGrid, "/geosemtwinx/traversability", 2)
        self.pub_goal = self.create_publisher(PoseStamped, "/geosemtwinx/nbv_goal", 2)
        self.pub_nbvpath = self.create_publisher(PathMsg, "/geosemtwinx/nbv_path", 2)
        self.pub_annot = self.create_publisher(Image, "/geosemtwinx/annotated", 2)
        self.pub_status = self.create_publisher(String, "/geosemtwinx/status", 2)
        self.tfb = TransformBroadcaster(self)
        self.create_service(Trigger, "/geosemtwinx/save_twin", self.on_save)
        self.create_timer(g("map_every_s"), self.publish_map)
        self.create_timer(g("assets_every_s"), self.publish_assets)
        self.create_timer(g("nbv_every_s"), self.publish_nbv)
        if g("dashboard_every_s") > 0:
            self.create_timer(g("dashboard_every_s"), lambda: self.save(final=False))
        self.path = PathMsg(); self.path.header.frame_id = g("map_frame")
        self.running = True
        self.worker = threading.Thread(target=self.loop, daemon=True); self.worker.start()
        self.get_logger().info(f"GeoSemTwin-X ready; waiting for {g('rgb_topic')} + {g('depth_topic')}")

    # ------------------------------------------------------------------ inputs
    def on_info(self, msg):
        if self.K is None:
            s = self.g("image_scale")
            self.K = np.array(msg.k, float).reshape(3, 3); self.K[:2] *= s
            self.size = (int(msg.width * s), int(msg.height * s))
            self.get_logger().info(f"camera: {msg.width}x{msg.height}, fx={msg.k[0]:.1f} (processing scale {s})")

    def on_init_pose(self, msg):
        if self.T_init is None or self.g("init_mode") == "topic":
            self.T_init = pose_to_matrix(msg.pose); self.get_logger().info("initial pose received")

    def on_rgbd(self, rgb_msg, depth_msg):
        if self.K is None or (self.g("init_mode") == "topic" and self.T_init is None):
            return
        try:
            rgb, depth = image_to_numpy(rgb_msg), image_to_numpy(depth_msg)
        except ValueError as e:
            self.get_logger().warn(str(e)); return
        s = self.g("image_scale")
        if s != 1.0:
            import cv2
            rgb = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, self.size, interpolation=cv2.INTER_NEAREST)
        if depth.shape != rgb.shape[:2]:
            self.get_logger().warn("depth and colour sizes differ: enable depth alignment on the camera", throttle_duration_sec=5)
            return
        self.queue.append((rgb_msg.header.stamp, dict(rgb=rgb, depth=depth)))

    # ------------------------------------------------------------------ processing
    def loop(self):
        while self.running:
            if not self.queue:
                time.sleep(0.005); continue
            stamp, frame = self.queue.popleft()
            with self.lock:
                if self.lt is None:
                    h, w = frame["depth"].shape
                    g = self.g
                    self.lt = LiveTwin(self.K, w, h, self.classes, detector=self.detector, T_init=self.T_init,
                                       detect_every=g("detect_every"), depth_max=g("depth_max"), map_voxel=g("map_voxel"),
                                       map_sigma_max=g("map_sigma_max"), map_extent=g("map_extent"), floor_z=g("floor_z"),
                                       floor_classes=g("floor_classes"), tracker=g("tracker"), nbv=g("init_mode") != "identity")
                t = time.time()
                r = self.lt.process(frame)
                dt = time.time() - t
            self.publish_pose(stamp, r["pose"])
            if r["keyframe"] and self.detector is not None and self.pub_annot.get_subscription_count():
                ann = draw_camera(frame["rgb"], r["detections"], self.classes, f"{len(r['detections'])} detections")
                self.pub_annot.publish(numpy_to_image(Image, ann[..., ::-1], "rgb8", stamp, self.g("camera_frame")))
            st = self.lt.stats(); st.update(last_frame_ms=1000 * dt, queue=len(self.queue), sigma=r["sigma"])
            self.pub_status.publish(String(data=json.dumps(st, default=float)))

    # ------------------------------------------------------------------ outputs
    def publish_pose(self, stamp, T):
        ps = PoseStamped(); ps.header.stamp, ps.header.frame_id = stamp, self.g("map_frame"); ps.pose = matrix_to_pose(Pose, T)
        self.pub_pose.publish(ps)
        self.path.header.stamp = stamp; self.path.poses.append(ps); self.pub_path.publish(self.path)
        tf = TransformStamped(); tf.header.stamp, tf.header.frame_id = stamp, self.g("map_frame")
        tf.child_frame_id = self.g("camera_frame")
        p = ps.pose
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = p.position.x, p.position.y, p.position.z
        tf.transform.rotation = p.orientation
        self.tfb.sendTransform(tf)

    def publish_map(self):
        if self.lt is None:
            return
        with self.lock:
            P, C = self.lt.map_points(150000)
            trav = self.lt.traversability() if self.lt.planner is not None else None
        now = self.get_clock().now().to_msg()
        self.pub_map.publish(make_pointcloud2(PointCloud2, PointField, P.astype(np.float32), C, now, self.g("map_frame")))
        if trav is not None:
            self.pub_grid.publish(traversability_grid(OccupancyGrid, trav, self.lt.occ, self.g("floor_z"), now, self.g("map_frame")))

    def publish_assets(self):
        if self.lt is None:
            return
        with self.lock:
            A = self.lt.assets()
        self.pub_assets.publish(asset_markers(MarkerArray, Marker, A, self.classes, self.get_clock().now().to_msg(),
                                              self.g("map_frame")))

    def publish_nbv(self):
        if self.lt is None or self.lt.planner is None or self.lt.i < 10:
            return
        with self.lock:
            plan = self.lt.plan_next_view()
        if not plan or "goal" not in plan:
            return
        now = self.get_clock().now().to_msg(); fz = self.g("floor_z")
        T = np.eye(4); c, s = np.cos(plan["yaw"]), np.sin(plan["yaw"])
        T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; T[:3, 3] = (plan["goal"][0], plan["goal"][1], fz)
        g = PoseStamped(); g.header.stamp, g.header.frame_id = now, self.g("map_frame"); g.pose = matrix_to_pose(Pose, T)
        self.pub_goal.publish(g)
        pm = PathMsg(); pm.header = g.header
        for x, y in plan["path"]:
            q = PoseStamped(); q.header = g.header; q.pose.position.x, q.pose.position.y, q.pose.position.z = float(x), float(y), fz
            q.pose.orientation.w = 1.0; pm.poses.append(q)
        self.pub_nbvpath.publish(pm)
        self.get_logger().info(f"next-best view -> ({plan['goal'][0]:.1f}, {plan['goal'][1]:.1f}) gains {plan['gains']}")

    def save(self, final=True):
        if self.lt is None:
            return "nothing to save yet"
        out = Path(self.g("output_dir")); out.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if final:
                twin = self.lt.finish(store_dir=out / "twin_store", dashboard_path=out / "Live_Dashboard.html",
                                      session="ros-live", title="GeoSemTwin-X live twin (ROS 2)")
            else:
                from geosemtwinx.dashboard import build_dashboard
                from geosemtwinx.twin import DigitalTwin
                twin = DigitalTwin("GeoSemTwin-X live twin (ROS 2)")
                twin.update(self.lt.gasf.finalize(), self.classes, "ros-live", occ_new=self.lt.occ)
                P, C = self.lt.map_points()
                build_dashboard(out / "Live_Dashboard.html", twin, P, C, traj_est=np.array(self.lt.traj),
                                perf=dict(stages_ms=self.lt.stats()["ms"]), title="GeoSemTwin-X live twin (ROS 2)")
        return f"saved {len(twin.present())} assets to {out}"

    def on_save(self, req, resp):
        resp.message = self.save(final=False); resp.success = True
        return resp

    def destroy_node(self):
        self.running = False
        try:
            self.get_logger().info(self.save(final=True))
        except Exception as e:
            print("save on shutdown failed:", e)
        super().destroy_node()


def main():
    rclpy.init()
    node = GeoSemTwinXNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

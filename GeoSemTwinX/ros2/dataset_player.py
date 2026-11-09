#!/usr/bin/env python3
"""Publishes a dataset as a live RGB-D camera so the whole system can be demonstrated in ROS 2 / RViz
without a robot.  dataset: staer | tum | synthetic

Topics: /dataset/rgb (rgb8), /dataset/depth (32FC1, metres), /dataset/camera_info, /dataset/gt_path (Path, latched),
        /geosemtwinx/initial_pose (PoseStamped, latched: first ground-truth pose -> gravity-aligned world)
  python3 ros2/dataset_player.py --ros-args -p dataset:=staer -p rate:=5.0
"""
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from ros_utils import matrix_to_pose, numpy_to_image  # noqa: E402


class Player(Node):
    def __init__(self):
        super().__init__("dataset_player")
        P = self.declare_parameter
        P("dataset", "staer"); P("scene", "scene03"); P("walk", "001"); P("tum_sequence", "rgbd_dataset_freiburg1_desk")
        P("data_root", str(ROOT / "data")); P("rate", 5.0); P("stride", 1); P("loop", False); P("frame_id", "camera_optical")
        P("max_frames", 0)
        g = lambda n: self.get_parameter(n).value
        ds, root = g("dataset"), g("data_root")
        mf = g("max_frames") or None
        if ds == "staer":
            from geosemtwinx.data import StaerSequence
            self.seq = StaerSequence(g("scene"), g("walk"), root=f"{root}/staer", scale=0.5, stride=g("stride"), max_frames=mf)
        elif ds == "tum":
            from geosemtwinx.data import TUMSequence
            self.seq = TUMSequence(g("tum_sequence"), root=f"{root}/tum", stride=g("stride"), max_frames=mf)
        else:
            from geosemtwinx.data import SyntheticSequence
            self.seq = SyntheticSequence("A", 720, stride=max(2, g("stride")), scale=1.0)
        self.frame_id, self.loop = g("frame_id"), g("loop")
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_rgb = self.create_publisher(Image, "/dataset/rgb", 10)
        self.pub_depth = self.create_publisher(Image, "/dataset/depth", 10)
        self.pub_info = self.create_publisher(CameraInfo, "/dataset/camera_info", 10)
        self.pub_gt = self.create_publisher(PathMsg, "/dataset/gt_path", latched)
        self.pub_init = self.create_publisher(PoseStamped, "/geosemtwinx/initial_pose", latched)
        self.i = 0
        f0 = self.seq.frame(0); self.h, self.w = f0["depth"].shape
        if self.seq.gt_poses is not None and ds != "tum":            # TUM ground truth is not gravity-aligned to a floor
            init = PoseStamped(); init.header.frame_id = "map"; init.pose = matrix_to_pose(Pose, self.seq.gt_poses[0])
            self.pub_init.publish(init)
        if self.seq.gt_poses is not None:
            path = PathMsg(); path.header.frame_id = "map"
            T0 = np.eye(4) if ds != "tum" else np.linalg.inv(self.seq.gt_poses[0])
            for T in self.seq.gt_poses:
                ps = PoseStamped(); ps.header.frame_id = "map"; ps.pose = matrix_to_pose(Pose, T0 @ T); path.poses.append(ps)
            self.pub_gt.publish(path)
        self.create_timer(1.0 / g("rate"), self.tick)
        self.get_logger().info(f"playing {ds}: {len(self.seq)} frames at {g('rate')} Hz ({self.w}x{self.h})")

    def tick(self):
        if self.i >= len(self.seq):
            if not self.loop:
                return
            self.i = 0
        f = self.seq.frame(self.i)
        stamp = self.get_clock().now().to_msg()
        K = self.seq.K
        info = CameraInfo(); info.header.stamp, info.header.frame_id = stamp, self.frame_id
        info.width, info.height = self.w, self.h
        info.k = [float(v) for v in K.ravel()]
        info.p = [float(K[0, 0]), 0.0, float(K[0, 2]), 0.0, 0.0, float(K[1, 1]), float(K[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; info.distortion_model = "plumb_bob"; info.d = [0.0] * 5
        self.pub_info.publish(info)
        self.pub_rgb.publish(numpy_to_image(Image, f["rgb"], "rgb8", stamp, self.frame_id))
        self.pub_depth.publish(numpy_to_image(Image, f["depth"].astype(np.float32), "32FC1", stamp, self.frame_id))
        self.i += 1
        if self.i % 50 == 0:
            self.get_logger().info(f"frame {self.i}/{len(self.seq)}")


def main():
    rclpy.init()
    n = Player()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()

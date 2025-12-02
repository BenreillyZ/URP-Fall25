#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, TransformException

# 这一行很关键：导入后会给 Buffer.register PoseStamped 的 transform 支持
import tf2_geometry_msgs  # noqa: F401


class ArucoToTFNode(Node):
    """
    订阅 /aruco_tracker/pose (在相机光学坐标系下),
    通过 TF 转到 target_frame (默认 base_link)，
    同时广播一个 aruco_tag_xx 的 TF，方便在 RViz 里看。
    """

    def __init__(self):
        super().__init__("aruco_to_tf_node")

        # 参数：目标父坐标系 & ArUco 的 child frame 名字
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("marker_frame", "aruco_tag_26")
        self.declare_parameter("input_topic", "/aruco_tracker/pose")
        self.declare_parameter("output_topic", "/aruco_in_base_link")

        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        self.marker_frame = (
            self.get_parameter("marker_frame").get_parameter_value().string_value
        )
        self.input_topic = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        self.output_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )

        # TF buffer + listener + broadcaster
        # 给个 cache_time，随便 10s 就够用
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        # spin_thread=True 会在内部开线程拉 TF，不堵你的 node
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_broadcaster = TransformBroadcaster(self)

        # 订阅 ArUco Pose（相机光学坐标下）
        self.pose_sub = self.create_subscription(
            PoseStamped, self.input_topic, self.pose_callback, 10
        )

        # 发布 ArUco 在 base_link 下的 Pose
        self.pose_pub = self.create_publisher(PoseStamped, self.output_topic, 10)

        self.get_logger().info(
            f"ArucoToTFNode started.\n"
            f"  input_topic  : {self.input_topic}\n"
            f"  output_topic : {self.output_topic}\n"
            f"  target_frame : {self.target_frame}\n"
            f"  marker_frame : {self.marker_frame}"
        )

    def pose_callback(self, msg: PoseStamped):
        """
        收到 /aruco_tracker/pose:
        - msg.header.frame_id = 相机光学坐标 (比如 camera_color_optical_frame)
        - msg.pose = ArUco 相对该坐标系的位姿
        """
        source_frame = msg.header.frame_id

        if not source_frame:
            self.get_logger().warn("收到的 PoseStamped 没有 frame_id，丢弃。")
            return

        # 直接调用 Buffer.transform：把 PoseStamped 从 source_frame -> target_frame
        try:
            aruco_in_target: PoseStamped = self.tf_buffer.transform(
                msg,
                self.target_frame,                  # 目标坐标系，比如 base_link
                timeout=Duration(seconds=0.2),      # 最长等 0.2s
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"TF transform failed: {source_frame} -> {self.target_frame}: {ex}"
            )
            return

        # 发布转换后的 PoseStamped（frame_id 已经是 target_frame）
        self.pose_pub.publish(aruco_in_target)

        # 再广播一个 TF： target_frame -> marker_frame
        t = TransformStamped()
        t.header.stamp = aruco_in_target.header.stamp
        t.header.frame_id = self.target_frame
        t.child_frame_id = self.marker_frame

        t.transform.translation.x = aruco_in_target.pose.position.x
        t.transform.translation.y = aruco_in_target.pose.position.y
        t.transform.translation.z = aruco_in_target.pose.position.z
        t.transform.rotation = aruco_in_target.pose.orientation

        self.tf_broadcaster.sendTransform(t)

        self.get_logger().debug(
            f"Published {self.marker_frame} in {self.target_frame}: "
            f"({aruco_in_target.pose.position.x:.3f}, "
            f"{aruco_in_target.pose.position.y:.3f}, "
            f"{aruco_in_target.pose.position.z:.3f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ArucoToTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("ArucoToTFNode shutting down (Ctrl+C).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

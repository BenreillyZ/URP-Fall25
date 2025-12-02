#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2

def get_aruco_dict():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

def get_aruco_detector(aruco_dict):
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(aruco_dict, parameters)
    else:
        # 老接口下返回 (dict, params) 二元组，调用 detectMarkers 时用
        return (aruco_dict, parameters)

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        # 只识别的 ArUco ID
        self.target_id = 26

        # 标签实际边长（米）——请按你的纸张尺寸改
        self.marker_length = 0.19

        # 订阅图像与相机内参
        self.image_sub = self.create_subscription(
            Image, '/sirar/realsense/color/image_raw', self.image_callback, 10
        )
        self.camerainfo_sub = self.create_subscription(
            CameraInfo, '/sirar/realsense/color/camera_info', self.camerainfo_callback, 10
        )

        # 发布 Pose
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_tracker/pose', 10)

        self.bridge = CvBridge()

        # 相机内参缓存（由 CameraInfo 动态更新）
        self.camera_matrix = None
        self.dist_coeffs = None
        self.frame_id = "camera_color_optical_frame"

        # ArUco 检测器
        self.aruco_dict = get_aruco_dict()
        self.aruco_detector = get_aruco_detector(self.aruco_dict)

        self.get_logger().info("ArucoDetectorNode started. Dict=Original ArUco, ID=26")

    def camerainfo_callback(self, msg: CameraInfo):
        # 将 CameraInfo 的 K(3x3) 与 D(N) 转为 OpenCV 需要的格式
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64).reshape(-1, 1)  # 列向量
        self.camera_matrix = K
        self.dist_coeffs = D
        # 记录 color optical frame（不同驱动可能不同）
        if msg.header.frame_id:
            self.frame_id = msg.header.frame_id

    def image_callback(self, msg: Image):
        if self.camera_matrix is None or self.dist_coeffs is None:
            # 还没收到 CameraInfo，先不做检测
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 兼容新老接口的 detectMarkers
        if isinstance(self.aruco_detector, tuple):
            # 老接口
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_detector[1]
            )
        else:
            # 新接口
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return

        # 只保留 ID == 26 的标记
        sel = [i for i, mid in enumerate(ids.flatten()) if int(mid) == self.target_id]
        if not sel:
            return

        # 估计位姿
        # 注意：estimatePoseSingleMarkers 需要 float 型相机参数
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[i] for i in sel],
            self.marker_length,
            self.camera_matrix.astype(np.float64),
            self.dist_coeffs.astype(np.float64),
        )

        # 逐个（通常就一个）发布 Pose
        for i in range(len(sel)):
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]

            rot_mtx, _ = cv2.Rodrigues(rvec)
            qx, qy, qz, qw = self.rotation_matrix_to_quaternion(rot_mtx)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = msg.header.stamp
            pose_msg.header.frame_id = self.frame_id

            pose_msg.pose.position.x = float(tvec[0])
            pose_msg.pose.position.y = float(tvec[1])
            pose_msg.pose.position.z = float(tvec[2])
            pose_msg.pose.orientation.x = float(qx)
            pose_msg.pose.orientation.y = float(qy)
            pose_msg.pose.orientation.z = float(qz)
            pose_msg.pose.orientation.w = float(qw)

            self.pose_pub.publish(pose_msg)

        self.get_logger().info(f"Published pose for ArUco ID {self.target_id}")

    @staticmethod
    def rotation_matrix_to_quaternion(R):
        # 返回 (x, y, z, w)
        q = np.empty((4,), dtype=np.float64)
        t = np.trace(R)
        if t > 0.0:
            t = np.sqrt(t + 1.0)
            q[3] = 0.5 * t
            t = 0.5 / t
            q[0] = (R[2, 1] - R[1, 2]) * t
            q[1] = (R[0, 2] - R[2, 0]) * t
            q[2] = (R[1, 0] - R[0, 1]) * t
        else:
            i = 0
            if R[1, 1] > R[0, 0]:
                i = 1
            if R[2, 2] > R[i, i]:
                i = 2
            j = (i + 1) % 3
            k = (j + 1) % 3
            t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
            q[i] = 0.5 * t
            t = 0.5 / t
            q[3] = (R[k, j] - R[j, k]) * t
            q[j] = (R[j, i] + R[i, j]) * t
            q[k] = (R[k, i] + R[i, k]) * t
        return q[0], q[1], q[2], q[3]

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

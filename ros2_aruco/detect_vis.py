#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
import os

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
        return (aruco_dict, parameters)

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        # 只识别的 ArUco ID
        self.target_id = 26
        # 你的实际标记边长（米）
        self.marker_length = 0.19

        # 订阅图像与相机内参（保持你的话题名）
        self.image_sub = self.create_subscription(
            Image, '/sirar/realsense/color/image_raw', self.image_callback, 10
        )
        self.camerainfo_sub = self.create_subscription(
            CameraInfo, '/sirar/realsense/color/camera_info', self.camerainfo_callback, 10
        )

        # 发布 Pose 与调试图像
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_tracker/pose', 10)
        self.debug_img_pub = self.create_publisher(Image, '/aruco_tracker/debug_image', 10)

        self.bridge = CvBridge()

        # 相机内参缓存
        self.camera_matrix = None
        self.dist_coeffs = None
        self.frame_id = "camera_color_optical_frame"

        # 可视化控制
        self.declare_parameter('show_window', True)
        self.declare_parameter('publish_debug_image', True)
        self.show_window = bool(self.get_parameter('show_window').value)
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)
        self.gui_ok = self.show_window and ('DISPLAY' in os.environ)
        if self.show_window and not self.gui_ok:
            self.get_logger().warn("未检测到 DISPLAY，关闭窗口模式，仅发布 debug_image 话题。")
            self.show_window = False

        if self.show_window:
            cv2.namedWindow('Aruco Debug', cv2.WINDOW_NORMAL)

        # 计算 FPS
        self._last_t = None
        self._fps = 0.0

        # ArUco 检测器
        self.aruco_dict = get_aruco_dict()
        self.aruco_detector = get_aruco_detector(self.aruco_dict)

        self.get_logger().info("ArucoDetectorNode started. Dict=Original ArUco, ID=26")

    def camerainfo_callback(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        self.camera_matrix = K
        self.dist_coeffs = D
        if msg.header.frame_id:
            self.frame_id = msg.header.frame_id

    def _estimate_poses(self, corner_list):
        """兼容：优先用 estimatePoseSingleMarkers；没有则回退 solvePnP。"""
        K = self.camera_matrix.astype(np.float64)
        D = self.dist_coeffs.astype(np.float64)
        if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corner_list, self.marker_length, K, D
            )
            r_list = [rv[0] for rv in rvecs]
            t_list = [tv[0] for tv in tvecs]
            return r_list, t_list

        # 回退：solvePnP（IPPE_SQUARE 更稳）
        s = self.marker_length / 2.0
        objp = np.array([[-s,  s, 0],
                         [ s,  s, 0],
                         [ s, -s, 0],
                         [-s, -s, 0]], dtype=np.float64)
        r_list, t_list = [], []
        for c in corner_list:
            imgp = c.reshape(-1, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            r_list.append(rvec.reshape(3))
            t_list.append(tvec.reshape(3))
        return r_list, t_list

    def image_callback(self, msg: Image):
        if self.camera_matrix is None or self.dist_coeffs is None:
            return

        # FPS 计算
        now = time.time()
        if self._last_t is not None:
            dt = max(1e-6, now - self._last_t)
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._last_t = now

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测标记
        if isinstance(self.aruco_detector, tuple):
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_detector[1]
            )
        else:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        vis = frame.copy()
        if ids is not None and len(ids) > 0:
            # 画出所有检测到的标记与 ID
            try:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            except Exception:
                pass

            # 仅选择 ID=26 的候选
            sel_idx = [i for i, mid in enumerate(ids.flatten()) if int(mid) == self.target_id]
            if sel_idx:
                sel_corners = [corners[i] for i in sel_idx]
                rvecs, tvecs = self._estimate_poses(sel_corners)

                for rvec, tvec in zip(rvecs, tvecs):
                    # 坐标轴（单位=marker_length）
                    try:
                        cv2.aruco.drawAxis(vis,
                                           self.camera_matrix.astype(np.float64),
                                           self.dist_coeffs.astype(np.float64),
                                           rvec, tvec, self.marker_length * 0.6)
                    except Exception:
                        pass

                    # 发布 Pose
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

                    # 叠加文字：距离与ID
                    dist = float(np.linalg.norm(tvec))
                    cv2.putText(vis, f"ID {self.target_id}  dist: {dist:.3f} m",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 左上角叠加 FPS
        cv2.putText(vis, f"FPS: {self._fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 发布调试图像
        if self.publish_debug_image:
            try:
                dbg_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
                dbg_msg.header = msg.header
                self.debug_img_pub.publish(dbg_msg)
            except Exception as e:
                self.get_logger().warn(f"publish debug image failed: {e}")

        # 本地窗口显示
        if self.show_window:
            try:
                cv2.imshow('Aruco Debug', vis)
                cv2.waitKey(1)
            except cv2.error as e:
                self.get_logger().warn(f"OpenCV GUI error: {e}. Disable window.")
                self.show_window = False
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    @staticmethod
    def rotation_matrix_to_quaternion(R):
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
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()

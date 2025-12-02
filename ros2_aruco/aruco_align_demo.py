#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


# ================== Kinematics: DH & Jacobian ==================

sira_dh_param = np.array([
    [0, 0.0,      0.1807,  np.pi/2],
    [0, -0.6127,  0.0,     0.0],
    [0, -0.57155, 0.0,     0.0],
    [0,  0.0,     0.17415, np.pi/2],
    [0,  0.0,     0.11985,-np.pi/2],
    [0,  0.0,     0.11655, 0.0],
], dtype=float)

# adjust for end-effector
sira_dh_param[5, 2] += 0.16  # +16cm


def transform_from_dh(theta, a, d, alpha):
    """标准 DH -> 4x4 齐次变换矩阵"""
    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,      d],
        [0.0,   0.0,     0.0,    1.0],
    ], dtype=float)


def fk_and_jacobian(q, dh=sira_dh_param):
    """
    用 DH 算当前末端位姿和 6x6 Jacobian
    q: [6,] 关节角（弧度）
    返回: eef_pos(3,), J(6,6)
    """
    q = np.asarray(q, dtype=float).reshape(6,)

    transforms = []
    transforms.append(transform_from_dh(q[0], dh[0, 1], dh[0, 2], dh[0, 3]))
    T06 = transforms[0]
    for i in range(1, dh.shape[0]):
        transforms.append(transform_from_dh(q[i], dh[i, 1], dh[i, 2], dh[i, 3]))
        T06 = T06 @ transforms[-1]

    eef_pos = T06[0:3, 3].copy()

    T0i = np.eye(4, dtype=float)
    J = np.zeros((6, 6), dtype=float)
    for i in range(dh.shape[0]):
        if i > 0:
            T0i = T0i @ transforms[i - 1]
        z_i = T0i[0:3, 2]
        p_i = T0i[0:3, 3]
        J[0:3, i] = np.cross(z_i, eef_pos - p_i)
        J[3:6, i] = z_i

    return eef_pos, J


# ================== Controller Node ==================


class ArucoAlignController(Node):
    """
    阶段 1：homing（先去到一个安全 home 姿态）
    阶段 2：用 Jacobian 控制末端对齐到 ArUco 上方 z_offset 米
    """

    def __init__(self):
        super().__init__("aruco_align_controller")

        # ---------- 参数 ----------
        self.declare_parameter("aruco_topic", "/aruco_in_base_link")
        # 注意：这里默认用 /joint_states，而不是 /sirar/ur10/joint_states
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("cmd_topic", "/forward_velocity_controller/commands")

        self.declare_parameter("z_offset", 0.10)        # 比标签高 10cm
        self.declare_parameter("kp_pos", 1.0)           # 位置 P 增益
        self.declare_parameter("max_lin_vel", 0.10)     # 最大线速度 m/s
        self.declare_parameter("max_joint_vel", 0.2)    # 最大关节速度 rad/s
        self.declare_parameter("control_rate", 125.0)   # 控制频率 Hz

        # homing 相关
        # !!! 这里请根据你的机器人实际安全姿态修改 !!!
        # 例子：简单用全 0，你可以改成 [0, -1.57, 1.57, 0, 1.57, 0] 之类
        self.declare_parameter(
            "q_home",
            [1.57085, -1.57083, -0.78539, 2.35622, 1.57084, -3.14155]
        )
        self.declare_parameter("kp_home", 0.8)          # 关节空间 P 增益
        self.declare_parameter("home_tol", 0.05)        # 关节距离阈值 rad

        self.aruco_topic = self.get_parameter("aruco_topic").value
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value

        self.z_offset = float(self.get_parameter("z_offset").value)
        self.kp_pos = float(self.get_parameter("kp_pos").value)
        self.max_lin_vel = float(self.get_parameter("max_lin_vel").value)
        self.max_joint_vel = float(self.get_parameter("max_joint_vel").value)
        self.control_rate = float(self.get_parameter("control_rate").value)

        self.q_home = np.array(self.get_parameter("q_home").value, dtype=float).reshape(6,)
        self.kp_home = float(self.get_parameter("kp_home").value)
        self.home_tol = float(self.get_parameter("home_tol").value)

        # ---------- 状态 ----------
        self.last_aruco_pose = None      # PoseStamped
        self.have_joints = False
        self.q = np.zeros(6, dtype=float)

        self.homing_done = False

        # ---------- 订阅 ----------
        self.sub_aruco = self.create_subscription(
            PoseStamped, self.aruco_topic, self.aruco_callback, 10
        )
        self.sub_js = self.create_subscription(
            JointState, self.joint_state_topic, self.joint_state_callback, 10
        )

        # ---------- 发布 ----------
        self.cmd_pub = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        # ---------- 定时器：主控制循环 ----------
        dt = 1.0 / self.control_rate
        self.timer = self.create_timer(dt, self.control_loop)

        self.get_logger().info(
            f"ArucoAlignController started.\n"
            f"  aruco_topic       : {self.aruco_topic}\n"
            f"  joint_state_topic : {self.joint_state_topic}\n"
            f"  cmd_topic         : {self.cmd_topic}\n"
            f"  z_offset          : {self.z_offset} m\n"
            f"  control_rate      : {self.control_rate} Hz\n"
            f"  q_home            : {self.q_home.tolist()}\n"
        )

    # ---------- Callbacks ----------

    def aruco_callback(self, msg: PoseStamped):
        self.last_aruco_pose = msg

    def joint_state_callback(self, msg: JointState):
        """
        把 joint_states 映射到我们 DH 使用的顺序:
        期望: [shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3]
        驱动发布顺序不同，所以按你原来的 RedundancyResolver 重排
        """
        if len(msg.position) < 6:
            return

        q = np.zeros(6, dtype=float)
        # 这部分保持和你原来代码一致
        q[0] = msg.position[5]
        q[1] = msg.position[3]
        q[2] = msg.position[0]
        q[3] = msg.position[1]
        q[4] = msg.position[4]
        q[5] = msg.position[2]

        self.q = q
        self.have_joints = True

    # ---------- 主控制循环 ----------

    def control_loop(self):
        # 先等到有 joint state
        if not self.have_joints:
            return

        # ========== 阶段 1：homing ==========
        if not self.homing_done:
            self.do_homing()
            return

        # ========== 阶段 2：视觉对齐 ==========
        if self.last_aruco_pose is None:
            # 没看到牌子就什么都不做，保持当前位置
            return

        try:
            p_eef, J = fk_and_jacobian(self.q)
        except Exception as e:
            self.get_logger().warn(f"FK/Jacobian failed: {e}")
            return

        # ArUco 在 base_link 下的位置
        p_a = np.array([
            self.last_aruco_pose.pose.position.x,
            self.last_aruco_pose.pose.position.y,
            self.last_aruco_pose.pose.position.z,
        ], dtype=float)

        # 目标点：比标签高 z_offset
        p_target = p_a + np.array([0.0, 0.0, self.z_offset], dtype=float)

        # 位置误差
        e_pos = p_target - p_eef
        dist = np.linalg.norm(e_pos)

        # 已经足够近 -> 停止
        if dist < 0.005:  # 5 mm
            cmd = Float64MultiArray(data=[0.0] * 6)
            self.cmd_pub.publish(cmd)
            self.get_logger().info("Reached target above ArUco (<= 5mm), sending zero velocity.")
            return

        # P 控制 + 限幅
        v = self.kp_pos * e_pos
        v_norm = np.linalg.norm(v)
        if v_norm > self.max_lin_vel:
            v *= self.max_lin_vel / max(v_norm, 1e-6)

        # 只控制位置，角速度先为 0
        u_cart = np.concatenate([v, np.zeros(3, dtype=float)], axis=0).reshape(6, 1)

        try:
            J_pinv = np.linalg.pinv(J)
            qdot = (J_pinv @ u_cart).reshape(6,)
        except Exception as e:
            self.get_logger().warn(f"Jacobian pinv failed: {e}")
            return

        # 关节速度限幅
        for i in range(6):
            if qdot[i] > self.max_joint_vel:
                qdot[i] = self.max_joint_vel
            elif qdot[i] < -self.max_joint_vel:
                qdot[i] = -self.max_joint_vel

        cmd = Float64MultiArray(data=qdot.tolist())
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"[Align] dist={dist:.3f} m, "
            f"eef=({p_eef[0]:.3f},{p_eef[1]:.3f},{p_eef[2]:.3f}), "
            f"target=({p_target[0]:.3f},{p_target[1]:.3f},{p_target[2]:.3f})"
        )

    # ---------- Homing 逻辑 ----------

    def do_homing(self):
        """
        关节空间 P 控制，把 q 拉到 q_home
        """
        e_q = self.q_home - self.q
        err_norm = np.linalg.norm(e_q)

        if err_norm < self.home_tol:
            # homing 完成 -> 发送零速度一次，标记 done
            self.homing_done = True
            cmd = Float64MultiArray(data=[0.0] * 6)
            self.cmd_pub.publish(cmd)
            self.get_logger().info(
                f"[Home] Done, |dq|={err_norm:.3f} < {self.home_tol:.3f}, "
                f"start visual alignment."
            )
            return

        # P 控制 + 限幅
        qdot = self.kp_home * e_q
        for i in range(6):
            if qdot[i] > self.max_joint_vel:
                qdot[i] = self.max_joint_vel
            elif qdot[i] < -self.max_joint_vel:
                qdot[i] = -self.max_joint_vel

        cmd = Float64MultiArray(data=qdot.tolist())
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"[Home] err_norm={err_norm:.3f}, q={self.q}, q_home={self.q_home}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = ArucoAlignController()
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

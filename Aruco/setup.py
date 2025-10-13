#!/usr/bin/env python3

import sys
import time
from threading import Event

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import moveit_commander
from geometry_msgs.msg import PoseStamped
from onrobot_rg_control.msg import OnRobotRGOutput  # 假设此消息定义在ROS 2工作区中是可用的

# 导入 tf2 相关库
import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import transform

# 保持常量不变
X_OFFSET = -0.01
Y_OFFSET = -0.04
TARGET_POSITION_OFFSET = 0.15

class AutomaticGripperController(Node):
    def __init__(self):
        # 1. 初始化 ROS 2 节点
        super().__init__('automatic_gripper_controller')

        # MoveIt! 初始化
        # 注意: moveit_commander.roscpp_initialize 必须在 rclpy.init 之后调用
        self.move_group = moveit_commander.MoveGroupCommander("manipulator")

        # 创建TF2的缓冲区和监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 创建夹爪指令的发布者
        self.gripper_pub = self.create_publisher(OnRobotRGOutput, '/OnRobotRGOutput', 10)

        # 用于等待ArUco位姿消息的变量和事件
        self.marker_pose = None
        self.pose_received_event = Event()
        
        self.get_logger().info("Automatic Gripper Controller Node has been started.")

    def gen_command(self, is_open):
        """生成夹爪的开合指令"""
        command = OnRobotRGOutput()
        max_width = 1600
        max_force = 400

        if is_open:
            command.rgfr = max_force  # ROS 2 消息字段通常是小写
            command.rgwd = max_width
            command.rctr = 16
        else:
            command.rgfr = max_force
            command.rgwd = 0
            command.rctr = 16
        return command

    def wait_for_marker_pose(self, timeout_sec=10.0):
        """订阅并等待一次ArUco标记位姿"""
        self.marker_pose = None
        self.pose_received_event.clear()
        
        # 创建一个一次性的订阅者
        sub = self.create_subscription(
            PoseStamped,
            '/aruco_tracker/pose',
            self._pose_callback,
            10
        )
        
        self.get_logger().info(f"Waiting for message on /aruco_tracker/pose for {timeout_sec} seconds...")
        # 等待事件被设置，或者超时
        event_is_set = self.pose_received_event.wait(timeout=timeout_sec)
        
        # 销毁订阅者，避免继续接收消息
        self.destroy_subscription(sub)

        if not event_is_set:
            self.get_logger().error("Timeout waiting for /aruco_tracker/pose message.")
            return None
        
        return self.marker_pose

    def _pose_callback(self, msg):
        """接收到消息的回调函数"""
        self.marker_pose = msg
        self.get_logger().info("Received ArUco marker pose.")
        self.pose_received_event.set()

    def run(self):
        """执行主要的业务逻辑"""
        # 保存初始位姿
        original_pose = self.move_group.get_current_pose().pose
        self.get_logger().info(f"Saved original pose: {original_pose.position}")

        # 等待并获取标记位姿
        marker_in_cam_pose = self.wait_for_marker_pose()
        if not marker_in_cam_pose:
            return # 获取失败，提前退出

        # 执行移动和抓取
        self.move_to_target_position(marker_in_cam_pose)
        
        # 等待一段时间以确保上一个动作完成
        time.sleep(1) 

        # 返回初始位置
        self.return_to_original_position(original_pose)

    def move_to_target_position(self, marker_in_cam_pose):
        """移动到目标位置并操作夹爪"""
        base_frame = "base"
        camera_frame = marker_in_cam_pose.header.frame_id

        try:
            # 1. 等待并执行坐标变换
            self.get_logger().info(f"Transforming pose from '{camera_frame}' to '{base_frame}'")
            transform_stamped = self.tf_buffer.lookup_transform(
                base_frame, camera_frame, rclpy.time.Time(), timeout=Duration(seconds=10.0)
            )
            marker_in_base_pose = transform(marker_in_cam_pose, transform_stamped)

            # 2. 计算目标位姿
            target_pose = PoseStamped()
            target_pose.header.frame_id = base_frame
            target_pose.pose = marker_in_base_pose.pose
            
            target_pose.pose.position.x += X_OFFSET
            target_pose.pose.position.y += Y_OFFSET
            target_pose.pose.position.z += TARGET_POSITION_OFFSET

            # 保持当前末端执行器的姿态
            current_orientation = self.move_group.get_current_pose().pose.orientation
            target_pose.pose.orientation = current_orientation

            # 3. 规划并移动
            self.move_group.set_pose_target(target_pose)
            plan_success = self.move_group.go(wait=True)
            self.move_group.stop() # 确保停止
            self.move_group.clear_pose_targets() # 清除目标

            if plan_success:
                self.get_logger().info("Robot moved to target position successfully!")
                
                # 4. 操作夹爪 (先打开，再闭合)
                # 打开
                gripper_command_open = self.gen_command(is_open=True)
                self.get_logger().info(f"Publishing gripper command to open: {gripper_command_open}")
                self.gripper_pub.publish(gripper_command_open)
                time.sleep(2.0) # 等待夹爪响应

                # 闭合
                gripper_command_close = self.gen_command(is_open=False)
                self.get_logger().info(f"Publishing gripper command to close: {gripper_command_close}")
                self.gripper_pub.publish(gripper_command_close)
                time.sleep(2.0) # 等待夹爪响应

            else:
                self.get_logger().error("Failed to move the robot to the target position.")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
            self.get_logger().error(f"TF Exception: {e}")

    def return_to_original_position(self, original_pose):
        """返回到初始位姿并打开夹爪"""
        self.get_logger().info("Returning to the original position.")
        self.move_group.set_pose_target(original_pose)
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        self.get_logger().info("Returned to the original position.")

        # 打开夹爪
        gripper_command = self.gen_command(is_open=True)
        self.get_logger().info(f"Gripper command to open: {gripper_command}")
        self.gripper_pub.publish(gripper_command)
        self.get_logger().info("Gripper opened after returning to the original position.")


def main(args=None):
    rclpy.init(args=args)
    # 必须在 rclpy.init() 之后初始化 moveit_commander
    moveit_commander.roscpp_initialize(sys.argv)

    try:
        controller_node = AutomaticGripperController()
        # 将节点的执行放在单独的线程或使用 MultiThreadedExecutor 可能更健壮
        # 但对于这个线性的脚本，直接调用 run() 也可以
        controller_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        # 确保按顺序关闭
        moveit_commander.roscpp_shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import sys
import time
from threading import Event

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import moveit_commander
from geometry_msgs.msg import PoseStamped
from onrobot_rg_control.msg import OnRobotRGOutput

# 导入 tf2 相关库
import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import transform

# 保持常量不变
X_OFFSET = 0.03
Y_OFFSET = -0.035
TARGET_POSITION_OFFSET = 0.17

class ComplexGripperController(Node):
    def __init__(self):
        super().__init__('complex_gripper_controller')

        # 状态变量
        self.is_gripper_opened = False

        # MoveIt! 初始化
        self.move_group = moveit_commander.MoveGroupCommander("manipulator")
        self.move_group.set_planning_time(10) # 增加规划时间

        # 创建TF2的缓冲区和监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 创建夹爪指令的发布者
        self.gripper_pub = self.create_publisher(OnRobotRGOutput, '/OnRobotRGOutput', 10)

        # 用于等待ArUco位姿消息
        self.marker_pose = None
        self.pose_received_event = Event()
        
        self.get_logger().info("Complex Gripper Controller Node has been started.")

    def gen_command(self, is_open):
        """生成夹爪的开合指令"""
        command = OnRobotRGOutput()
        max_width = 800  # 注意这个值与前一个脚本不同
        max_force = 400

        if is_open:
            command.rgfr = max_force
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
        
        sub = self.create_subscription(
            PoseStamped,
            '/aruco_tracker/pose',
            self._pose_callback,
            10
        )
        
        self.get_logger().info(f"Waiting for message on /aruco_tracker/pose for {timeout_sec} seconds...")
        event_is_set = self.pose_received_event.wait(timeout=timeout_sec)
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
        
    def move_to_pose(self, pose):
        """一个简单的封装函数，用于移动到指定位姿"""
        self.move_group.set_pose_target(pose)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        return success

    def run(self, target_offset_x=0.03):
        """执行主要的业务逻辑"""
        # 1. 保存初始位姿
        original_pose = self.move_group.get_current_pose().pose
        self.get_logger().info(f"Saved original pose: {original_pose.position}")

        # 2. 等待并获取标记位姿
        marker_in_cam_pose = self.wait_for_marker_pose()
        if not marker_in_cam_pose:
            return

        try:
            # 3. 坐标变换
            base_frame = "base"
            camera_frame = marker_in_cam_pose.header.frame_id
            self.get_logger().info(f"Transforming pose from '{camera_frame}' to '{base_frame}'")
            transform_stamped = self.tf_buffer.lookup_transform(
                base_frame, camera_frame, rclpy.time.Time(), timeout=Duration(seconds=10.0)
            )
            marker_in_base_pose = transform(marker_in_cam_pose, transform_stamped)

            # 4. 计算初始目标位姿
            target_pose = PoseStamped()
            target_pose.header.frame_id = base_frame
            target_pose.pose = marker_in_base_pose.pose
            target_pose.pose.position.x += X_OFFSET
            target_pose.pose.position.y += Y_OFFSET
            target_pose.pose.position.z += TARGET_POSITION_OFFSET
            target_pose.pose.orientation = self.move_group.get_current_pose().pose.orientation

            # 5. 移动到初始目标位姿
            self.get_logger().info("Moving to initial target position...")
            if not self.move_to_pose(target_pose):
                self.get_logger().error("Failed to move the robot to the target position.")
                return
            
            self.get_logger().info("Robot moved to target position successfully!")

            # 6. 执行序列操作
            if not self.is_gripper_opened:
                # 6.1 打开夹爪
                self.get_logger().info("Opening gripper...")
                self.gripper_pub.publish(self.gen_command(is_open=True))
                time.sleep(1.5)
                self.is_gripper_opened = True

                # 6.2 立即闭合夹爪
                self.get_logger().info("Closing gripper...")
                self.gripper_pub.publish(self.gen_command(is_open=False))
                time.sleep(1.5)

                # 6.3 返回原始位置
                self.get_logger().info("Returning to original position...")
                if not self.move_to_pose(original_pose):
                    self.get_logger().error("Failed to return to original position.")
                    return # 关键步骤失败，中止序列

                # 6.4 移动到X轴偏移后的新位置
                target_pose.pose.position.x += target_offset_x
                self.get_logger().info(f"Moving to offset position (X + {target_offset_x}m)...")
                if not self.move_to_pose(target_pose):
                    self.get_logger().error("Failed to move to the offset position.")
                    # 即使失败，也尝试返回原点
                    self.move_to_pose(original_pose)
                    return
                
                self.get_logger().info("Successfully moved to offset position.")

                # 6.5 再次打开夹爪（释放）
                self.get_logger().info("Opening gripper to release...")
                self.gripper_pub.publish(self.gen_command(is_open=True))
                time.sleep(1.5)

            else:
                self.get_logger().info("Gripper has been operated. Skipping sequence.")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
            self.get_logger().error(f"TF Exception: {e}")
        finally:
            # 7. 无论如何，最后都返回初始位置
            self.get_logger().info("Ensuring robot is at original position...")
            self.move_to_pose(original_pose)
            self.get_logger().info("Sequence finished. Robot is at original position.")


def main(args=None):
    rclpy.init(args=args)
    moveit_commander.roscpp_initialize(sys.argv)

    try:
        controller_node = ComplexGripperController()
        # 调用run方法执行整个序列
        controller_node.run(target_offset_x=0.03) # 传入X轴偏移量
    except KeyboardInterrupt:
        pass
    finally:
        moveit_commander.roscpp_shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
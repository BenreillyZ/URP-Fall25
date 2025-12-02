#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time


# 如果系统里没有 tf_transformations，可以用自写四元数/矩阵转换
import tf_transformations as tft

def pose_to_matrix(pose):
    t = tft.translation_matrix([pose.position.x, pose.position.y, pose.position.z])
    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    R = tft.quaternion_matrix(q)
    return t @ R

def mat_to_pose(T):
    p = PoseStamped().pose
    p.position.x, p.position.y, p.position.z = T[0, 3], T[1, 3], T[2, 3]
    q = tft.quaternion_from_matrix(T)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
    return p

def transform_to_matrix(tfmsg: TransformStamped):
    trans = [tfmsg.transform.translation.x, tfmsg.transform.translation.y, tfmsg.transform.translation.z]
    rot = [tfmsg.transform.rotation.x, tfmsg.transform.rotation.y, tfmsg.transform.rotation.z, tfmsg.transform.rotation.w]
    return tft.translation_matrix(trans) @ tft.quaternion_matrix(rot)

class TagGoalFromPose(Node):
    def __init__(self):
        super().__init__('tag_goal_from_pose')

        # 参数：可按需修改/重映射
        self.declare_parameter('base_frame', 'siraRbase')
        self.declare_parameter('tag_frame_name', 'aruco_tag')          # 广播出的tag帧名
        self.declare_parameter('tag_above_frame', 'aruco_tag_above')   # 广播出的“上方”帧名
        self.declare_parameter('hover_height_m', 0.10)                  # 距离tag的高度（沿tag本体+Z）
        self.declare_parameter('goal_topic', '/goal_above_tag')         # 输出的末端目标话题

        self.base_frame     = self.get_parameter('base_frame').value
        self.tag_frame_name = self.get_parameter('tag_frame_name').value
        self.tag_above_frame= self.get_parameter('tag_above_frame').value
        self.hover_height   = float(self.get_parameter('hover_height_m').value)
        self.goal_topic     = self.get_parameter('goal_topic').value

        # 订阅现有的 /aruco_tracker/pose
        self.pose_sub = self.create_subscription(PoseStamped, '/aruco_tracker/pose', self.pose_cb, 10)

        # 发布在 base_frame 下的目标位姿（可直接对接 MoveIt/自写控制器）
        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic, 10)

        # TF 缓存 & 广播器
       # self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))

        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info('TagGoalFromPose node started.')

    def pose_cb(self, msg: PoseStamped):
        """
        输入：/aruco_tracker/pose (PoseStamped)
        假设 msg.header.frame_id 是相机坐标系（如 realsense_color_optical_frame）
        """
        cam_frame = msg.header.frame_id
        try:
            # 查表：base -> camera 的变换
            tf_base_cam: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame=self.base_frame, source_frame=cam_frame,
                time=rclpy.time.Time(), timeout=Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(f'lookup_transform {self.base_frame} <- {cam_frame} failed: {e}')
            return

        # 构造矩阵
        T_base_cam = transform_to_matrix(tf_base_cam)
        T_cam_tag  = pose_to_matrix(msg.pose)
        T_base_tag = T_base_cam @ T_cam_tag

        # 广播：base -> aruco_tag（用于在TF树可视化）
        self.broadcast_tf(self.base_frame, self.tag_frame_name, T_base_tag, msg.header.stamp)

        # 计算“位于tag上方”的目标（沿 tag 的 +Z 方向抬高 hover_height）
        offset_tag = np.array([0.0, 0.0, self.hover_height, 1.0])      # tag自身+Z
        p_above_base_h = T_base_tag @ offset_tag                        # 齐次坐标
        T_base_tag_above = T_base_tag.copy()
        T_base_tag_above[:3, 3] = p_above_base_h[:3]

        # 这里的姿态：保持与 tag 相同（让工具坐标对齐tag）；也可按需求改为固定对齐世界Z等
        goal_pose = PoseStamped()
        goal_pose.header.stamp = msg.header.stamp
        goal_pose.header.frame_id = self.base_frame
        goal_pose.pose = mat_to_pose(T_base_tag_above)

        # 发布目标位姿
        self.goal_pub.publish(goal_pose)

        # 广播：base -> aruco_tag_above（用于在TF树可视化）
        self.broadcast_tf(self.base_frame, self.tag_above_frame, T_base_tag_above, msg.header.stamp)

    def broadcast_tf(self, parent, child, T, stamp):
        tfmsg = TransformStamped()
        tfmsg.header.stamp = stamp
        tfmsg.header.frame_id = parent
        tfmsg.child_frame_id = child
        tfmsg.transform.translation.x = float(T[0, 3])
        tfmsg.transform.translation.y = float(T[1, 3])
        tfmsg.transform.translation.z = float(T[2, 3])
        q = tft.quaternion_from_matrix(T)
        tfmsg.transform.rotation.x = float(q[0])
        tfmsg.transform.rotation.y = float(q[1])
        tfmsg.transform.rotation.z = float(q[2])
        tfmsg.transform.rotation.w = float(q[3])
        self.br.sendTransform(tfmsg)

def main():
    rclpy.init()
    node = TagGoalFromPose()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

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

# Import tf2 libraries
import tf2_ros
from tf2_geometry_msgs.tf2_geometry_msgs import transform

# Constants remain the same
X_OFFSET = 0.03
Y_OFFSET = -0.035
TARGET_POSITION_OFFSET = 0.17

class YAxisOffsetController(Node):
    def __init__(self):
        super().__init__('y_axis_offset_controller')

        # State variable
        self.is_gripper_opened = False

        # Initialize MoveIt! Commander
        self.move_group = moveit_commander.MoveGroupCommander("manipulator")
        self.move_group.set_planning_time(10) # Give it more time to find a plan

        # Initialize TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create publisher for the gripper
        self.gripper_pub = self.create_publisher(OnRobotRGOutput, '/OnRobotRGOutput', 10)

        # Variables for waiting on the ArUco pose message
        self.marker_pose = None
        self.pose_received_event = Event()
        
        self.get_logger().info("Y-Axis Offset Controller Node has been started.")

    def gen_command(self, is_open):
        """Generates the command for the OnRobot RG gripper."""
        command = OnRobotRGOutput()
        max_width = 800
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
        """Creates a one-time subscriber to wait for the ArUco marker pose."""
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
        """Callback for the one-time subscriber."""
        self.marker_pose = msg
        self.get_logger().info("Received ArUco marker pose.")
        self.pose_received_event.set()
        
    def move_to_pose(self, pose):
        """Helper function to plan and move to a target pose."""
        self.move_group.set_pose_target(pose)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        return success

    def run(self, target_offset_y=-0.03):
        """Executes the main sequence of operations."""
        # 1. Store the starting pose
        original_pose = self.move_group.get_current_pose().pose
        self.get_logger().info(f"Saved original pose: {original_pose.position}")

        # 2. Wait for the ArUco marker pose
        marker_in_cam_pose = self.wait_for_marker_pose()
        if not marker_in_cam_pose:
            return

        try:
            # 3. Transform the marker pose to the robot's base frame
            base_frame = "base"
            camera_frame = marker_in_cam_pose.header.frame_id
            self.get_logger().info(f"Transforming pose from '{camera_frame}' to '{base_frame}'")
            transform_stamped = self.tf_buffer.lookup_transform(
                base_frame, camera_frame, rclpy.time.Time(), timeout=Duration(seconds=10.0)
            )
            marker_in_base_pose = transform(marker_in_cam_pose, transform_stamped)

            # 4. Calculate the initial target pose above the marker
            target_pose = PoseStamped()
            target_pose.header.frame_id = base_frame
            target_pose.pose = marker_in_base_pose.pose
            target_pose.pose.position.x += X_OFFSET
            target_pose.pose.position.y += Y_OFFSET
            target_pose.pose.position.z += TARGET_POSITION_OFFSET
            target_pose.pose.orientation = self.move_group.get_current_pose().pose.orientation

            # 5. Move to the initial target pose
            self.get_logger().info("Moving to initial target position...")
            if not self.move_to_pose(target_pose):
                self.get_logger().error("Failed to move the robot to the target position.")
                return
            
            self.get_logger().info("Robot moved to target position successfully!")

            # 6. Execute the main gripper and movement sequence
            if not self.is_gripper_opened:
                # 6.1 Open gripper
                self.get_logger().info("Opening gripper...")
                self.gripper_pub.publish(self.gen_command(is_open=True))
                time.sleep(1.5)
                self.is_gripper_opened = True

                # 6.2 Close gripper
                self.get_logger().info("Closing gripper...")
                self.gripper_pub.publish(self.gen_command(is_open=False))
                time.sleep(1.5)

                # 6.3 Return to home position
                self.get_logger().info("Returning to original position...")
                if not self.move_to_pose(original_pose):
                    self.get_logger().error("Failed to return to original position.")
                    return # Critical step failed, abort sequence

                # 6.4 Move to the new position with a Y-axis offset
                # THIS IS THE KEY CHANGE FROM THE PREVIOUS SCRIPT
                target_pose.pose.position.y += target_offset_y
                self.get_logger().info(f"Moving to offset position (Y + {target_offset_y}m)...")
                if not self.move_to_pose(target_pose):
                    self.get_logger().error("Failed to move to the offset position.")
                    self.move_to_pose(original_pose) # Attempt to return home anyway
                    return
                
                self.get_logger().info("Successfully moved to offset position.")

                # 6.5 Open the gripper again (to release)
                self.get_logger().info("Opening gripper to release...")
                self.gripper_pub.publish(self.gen_command(is_open=True))
                time.sleep(1.5)

            else:
                self.get_logger().info("Gripper has been operated. Skipping sequence.")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
            self.get_logger().error(f"TF Exception: {e}")
        finally:
            # 7. Final step: ensure the robot returns to its original position
            self.get_logger().info("Ensuring robot is at original position...")
            self.move_to_pose(original_pose)
            self.get_logger().info("Sequence finished. Robot is at original position.")


def main(args=None):
    rclpy.init(args=args)
    # MoveIt Commander initialization must happen after rclpy.init()
    moveit_commander.roscpp_initialize(sys.argv)

    try:
        controller_node = YAxisOffsetController()
        # Call the run method to execute the sequence
        controller_node.run(target_offset_y=-0.03) # Provide the desired Y-axis offset
    except KeyboardInterrupt:
        pass
    finally:
        # Gracefully shut down
        moveit_commander.roscpp_shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
# UR5e ArUco Detection & Velocity Control (ROS 2)

This repository guides you through operating a UR5e robot with ArUco marker detection using ROS 2.

## 1. ArUco Detection
Launch the script to detect markers via the camera.

```bash
# Launch ArUco detect script
python3 /home/sirar/UR5e-Robot-Autonomously-Picks-Aruco-Markers-ROS-Noetic/ros2_aruco/detect.py
```

## 2. TF Tree Transformation
Unify the ArUco frame with the robot's TF tree.

```bash
# Deactivate conda (if applicable) to avoid environment conflicts
conda deactivate

# OPTIONAL: View current frames
ros2 run tf2_tools view_frames

# Publish static transform (base_link -> camera frame)
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link realsense_color_optical_frame

# Broadcast ArUco to TF
python3 /home/sirar/UR5e-Robot-Autonomously-Picks-Aruco-Markers-ROS-Noetic/ros2_aruco/aruco_to_tf_broadcaster.py
```

## 3. Robot Dashboard Control
Manage the robot's state using service calls.

```bash
# Play (Start robot)
ros2 service call /dashboard_client/play std_srvs/srv/Trigger

# Stop robot
ros2 service call /dashboard_client/stop std_srvs/srv/Trigger
```

## 4. Controller Management
Switch to the velocity controller.

```bash
# Switch to forward_velocity_controller
ros2 service call /controller_manager/switch_controller \
  controller_manager_msgs/srv/SwitchController \
  "{activate_controllers: ['forward_velocity_controller'], \
    deactivate_controllers: ['scaled_joint_trajectory_controller'], \
    strictness: 1}"
```

## 5. Send Velocity Commands
Publish velocity data directly.

```bash
# Example: Move joint 6 at 0.05 speed
ros2 topic pub /forward_velocity_controller/commands std_msgs/msg/Float64MultiArray "data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.05]"
```

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
import json
import sys
import argparse
from typing import Dict, List, Any, Tuple

# --- CONFIGURATION ---
# IMPORTANT: Update this path to where you saved the JSON file
TRAJECTORY_FILE = 'human_trajectories_50.json'
# Publish rate (Hz)
PUBLISH_RATE = 10

def parse_arguments():
    parser = argparse.ArgumentParser(description='Play back human motion trajectories')
    parser.add_argument('--scenario', '-s', type=int, default=0,
                      help='Scenario index to play (default: 0)')
    parser.add_argument('--list', '-l', action='store_true',
                      help='List all available scenarios and exit')
    parser.add_argument('--file', '-f', type=str, default=TRAJECTORY_FILE,
                      help=f'Path to trajectory file (default: {TRAJECTORY_FILE})')
    parser.add_argument('--offset-x', type=float, default=0.0,
                      help='Shift human position along X axis in meters (default: 0.0)')
    parser.add_argument('--offset-y', type=float, default=0.0,
                      help='Shift human position along Y axis in meters (default: 0.0)')
    parser.add_argument('--offset-z', type=float, default=0.0,
                      help='Shift human position along Z axis in meters (default: 0.0)')
    return parser.parse_args()

# Joint names for marker publishing (must match the original definition)
JOINT_NAMES = ['CLAV', 'C7', 'RSHO', 'LSHO', 'LAEL', 'RAEL', 'LWPS', 'RWPS',
               'L3', 'LHIP', 'RHIP', 'LKNE', 'RKNE', 'LHEE', 'RHEE']
NUM_JOINTS = len(JOINT_NAMES)


class TrajectoryPlaybackPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_playback_publisher')

        # Get command line arguments
        self.args = parse_arguments()

        # Position offset in meters
        self.offset_m = np.array([self.args.offset_x, self.args.offset_y, self.args.offset_z])
        
        # Log the offset being used
        if np.any(self.offset_m != 0):
            self.get_logger().info(f"Using position offset: [{self.offset_m[0]:.3f}, {self.offset_m[1]:.3f}, {self.offset_m[2]:.3f}] meters")

        # Publishers
        self.array_publisher = self.create_publisher(Float32MultiArray, 'joint_array', 10)
        self.marker_publisher = self.create_publisher(MarkerArray, 'joint_markers', 10)

        # State Variables
        self.all_trajectories: List[Dict[str, Any]] = self._load_trajectories()
        self.current_trajectory: List[Dict[str, Any]] = []
        self.current_frame_index = 0

        # Timer for playback
        self.timer = self.create_timer(1.0 / PUBLISH_RATE, self.timer_callback)

    def _load_trajectories(self) -> List[Dict[str, Any]]:
        """Loads all trajectories from the JSON file."""
        try:
            with open(self.args.file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.get_logger().error(f"Error: Trajectory file not found at {TRAJECTORY_FILE}")
            return []
        except json.JSONDecodeError:
            self.get_logger().error(f"Error: Failed to decode JSON from {TRAJECTORY_FILE}")
            return []

    def load_scenario(self, index: int):
        """Sets the current trajectory for playback."""
        if 0 <= index < len(self.all_trajectories):
            self.current_trajectory = self.all_trajectories[index]['trajectory']
            self.current_frame_index = 0
        else:
            self.get_logger().error(f"Scenario index {index} is out of bounds.")
            self.current_trajectory = []

    def timer_callback(self):
        """Reads the next frame and publishes the data and markers."""
        if not self.current_trajectory:
            self.get_logger().warn("No trajectory loaded or file error. Stopping playback.")
            self.timer.cancel()
            return

        if self.current_frame_index >= len(self.current_trajectory):
            # self.get_logger().info("Trajectory finished. Looping to start.")
            self.current_frame_index = 0
            # Optional: Uncomment the next line to stop at the end instead of looping
            # self.timer.cancel()
            # return

        # 1. Extract and Publish Float32MultiArray
        frame_data: Dict[str, Any] = self.current_trajectory[self.current_frame_index]
        joint_poses_mm: List[List[float]] = frame_data['joint_poses_mm']

        # Convert list-of-lists (mm) to the required numpy array (m) for the planner
        # Note: The planner is likely expecting meters for consistency with Gazebo
        joint_array_m = np.array(joint_poses_mm, dtype=np.float32) / 1000.0
        
        # Apply position offset (in meters)
        joint_array_m += self.offset_m

        # TODO Send the markers N timesteps late, to simulate this being "predictive"
        msg = Float32MultiArray()
        msg.data = joint_array_m.flatten().tolist()
        self.array_publisher.publish(msg)

        # 2. Publish Visualization Markers (with offset applied)
        joint_poses_mm_offset = [[x + self.offset_m[0]*1000, y + self.offset_m[1]*1000, z + self.offset_m[2]*1000] 
                                  for x, y, z in joint_poses_mm]
        self.publish_markers(joint_poses_mm_offset)

        self.get_logger().debug(
            f"Published frame {self.current_frame_index}/{len(self.current_trajectory)} at t={frame_data['time_s']:.2f}s")
        self.current_frame_index += 1

    # --- Marker Publishing Logic (Copied and adapted from posecap_pub.py) ---
    def to_point(self, xyz: Tuple[float, float, float]) -> Point:
        """Converts mm coordinates to a geometry_msgs/Point in meters."""
        pt = Point()
        pt.x = xyz[0] / 1000.0
        pt.y = xyz[1] / 1000.0
        pt.z = xyz[2] / 1000.0
        return pt

    def publish_markers(self, joint_xyz_mm: List[List[float]]):
        """Publishes the joint poses as RVIZ markers."""
        marker_array = MarkerArray()
        frame_id = "base"  # Use the robot's base frame for visualization

        # Publish joints as spheres
        for idx, (x, y, z) in enumerate(joint_xyz_mm):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "joints"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x / 1000.0  # Convert to meters
            marker.pose.position.y = y / 1000.0
            marker.pose.position.z = z / 1000.0
            # ... (rest of sphere marker properties: orientation, scale, color)
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        # Publish bones (links) as lines
        connections: List[Tuple[str, str]] = \
            [
                ('CLAV', 'C7'), ('C7', 'LSHO'), ('C7', 'RSHO'),
                ('LSHO', 'LAEL'), ('LAEL', 'LWPS'),
                ('RSHO', 'RAEL'), ('RAEL', 'RWPS'),
                ('C7', 'L3'), ('L3', 'LHIP'), ('L3', 'RHIP'),
                ('LHIP', 'LKNE'), ('LKNE', 'LHEE'),
                ('RHIP', 'RKNE'), ('RKNE', 'RHEE')
            ]
        name_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(JOINT_NAMES)}

        link_marker = Marker()
        link_marker.header.frame_id = frame_id
        link_marker.header.stamp = self.get_clock().now().to_msg()
        link_marker.ns = "links"
        link_marker.id = 1000
        link_marker.type = Marker.LINE_LIST
        link_marker.action = Marker.ADD
        link_marker.scale.x = 0.02
        link_marker.color.r = 0.0
        link_marker.color.g = 1.0

        link_marker.color.b = 0.0
        link_marker.color.a = 1.0

        for joint1_name, joint2_name in connections:
            if joint1_name in name_to_idx and joint2_name in name_to_idx:
                p1_mm = joint_xyz_mm[name_to_idx[joint1_name]]
                p2_mm = joint_xyz_mm[name_to_idx[joint2_name]]
                link_marker.points.append(self.to_point(p1_mm))
                link_marker.points.append(self.to_point(p2_mm))

        marker_array.markers.append(link_marker)
        self.marker_publisher.publish(marker_array)


def list_scenarios(trajectories):
    print("\nAvailable scenarios:")
    print("------------------")
    for idx, traj in enumerate(trajectories):
        print(f"{idx}: {traj['scenario_id']}")
        if 'description' in traj:
            print(f"   {traj['description']}")
    print()

def main(args=None):
    # Parse command line arguments
    parsed_args = parse_arguments()
    
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create node
    node = TrajectoryPlaybackPublisher()
    
    # If --list flag is used, show available scenarios and exit
    if parsed_args.list:
        list_scenarios(node.all_trajectories)
        node.destroy_node()
        rclpy.shutdown()
        return
    
    # Validate scenario index
    if parsed_args.scenario < 0 or parsed_args.scenario >= len(node.all_trajectories):
        print(f"Error: Scenario index {parsed_args.scenario} is out of range. Use --list to see available scenarios.")
        node.destroy_node()
        rclpy.shutdown()
        return
    
    # Load the specified scenario
    node.load_scenario(parsed_args.scenario)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
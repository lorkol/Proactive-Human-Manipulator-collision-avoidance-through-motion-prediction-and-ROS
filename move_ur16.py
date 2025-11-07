#!/usr/bin/env python3
from typing import Dict, List

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import cupy as cp
from cupy.typing import NDArray
from sensor_msgs.msg import JointState
from rclpy.executors import MultiThreadedExecutor
robot_joint_angles: cp.ndarray = cp.zeros(6)

class JointStateReader(Node):
    """Regarding the ROBOT joint positions."""
    def __init__(self) -> None:
        super().__init__('joint_state_reader')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)

    def joint_callback(self, msg) -> None:
        global robot_joint_angles
        joint_order: List[str] = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        '''Robot joint names in order'''
        j_positions: Dict[str, float] = dict(zip(msg.name, msg.position))
        robot_joint_angles = cp.array([j_positions[joint] for joint in joint_order])
        

class UR16TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('ur16_destination_loop')
        self.publisher_ = self.create_publisher(
            JointTrajectoryPoint,
            '/goal_pose',
            10
        )
        self.current_goal: NDArray = None
        self.time_window: int = 1*1000000000  # 1 second in nanoseconds
        self.step = 1
        self.timer = self.create_timer(1, self.timer_callback)
        self.send_goal([0.0, -1.0, 1.0, -3.0, 0.0, 0.0])
        
    def send_goal(self, position: List[float]):
        traj = JointTrajectory()
        traj.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
        ]
        point = JointTrajectoryPoint()
        point.positions = position
        point.time_from_start = Duration(nanosec=self.time_window)
        self.current_goal = cp.array(position)
        self.publisher_.publish(point)
        self.get_logger().info(f'Step {self.step}: Published point {position}')

    def timer_callback(self):
        if cp.linalg.norm(((robot_joint_angles - self.current_goal + cp.pi) % (2 * cp.pi)) - cp.pi) > 0.1:
            # self.get_logger().info("Not reached yet.")
            # self.get_logger().info(f"current: {robot_joint_angles}, goal: {self.current_goal}, norm: {cp.linalg.norm(robot_joint_angles - self.current_goal)}")
            return
        if self.step == 0:
            # Step 0: Move to left above object
            self.send_goal([0.0, -1.0, 1.0, -3.0, 0.0, 0.0])
        elif self.step == 1:
            # Step 1: Drop down to pick
            self.send_goal([-2.0, -2.0, 1.0, -3.0, 0.0, 0.0])
        elif self.step == 2:
            # Step 2: Move up
            self.send_goal([0.0, -2.0, 1.0, -3.0, 0.0, 0.0])
        elif self.step == 3:
            # Step 3: Move to right above target
            self.send_goal([0.0, -2.0, 1.0, -3.0, 0.0, 0.0])
        elif self.step == 4:
            # Step 4: Drop down to place
            self.send_goal([0.0, -2.0, 1.0, -3.0, 0.0, 0.0])
        elif self.step == 5:
            # Step 5: Move up
            self.send_goal([-1.0, -2.0, 1.0, -1.0, 1.0, 0.0])
        elif self.step == 6:
            # Step 6: Move back to left
            self.send_goal([1.0, -2.0, 1.0, -1.0, 1.0, 0.0])

        self.step = (self.step + 1) % 7

def main(args=None):
    rclpy.init(args=args)
    joint_reader = JointStateReader()
    trajectory_publisher = UR16TrajectoryPublisher()

    executor = MultiThreadedExecutor()
    executor.add_node(joint_reader)
    executor.add_node(trajectory_publisher)

    try:
        executor.spin()
    finally:
        joint_reader.destroy_node()
        trajectory_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

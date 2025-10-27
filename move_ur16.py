#!/usr/bin/env python3
from typing import List

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class UR10TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('ur10_pick_place_loop')
        self.publisher_ = self.create_publisher(
            JointTrajectoryPoint,
            '/joint_destination',
            10
        )
        self.timer = self.create_timer(3, self.timer_callback)
        self.step = 0
        
    def send_goal(self, position: List[float], duration_sec: float):
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
        point.time_from_start = Duration(nanosec=duration_sec)
        self.publisher_.publish(point)
        self.get_logger().info(f'Step {self.step}: Published point {position}')

    def timer_callback(self):
        t=1*1000000000
        if self.step == 0:
            # Step 0: Move to left above object
            self.send_goal([0.0, -1.0, 1.0, -3.0, 0.0, 0.0], t)
        elif self.step == 1:
            # Step 1: Drop down to pick
            self.send_goal([-2.0, -2.0, 1.0, -3.0, 0.0, 0.0], t)
        elif self.step == 2:
            # Step 2: Move up
            self.send_goal([0.0, -2.0, 1.0, -3.0, 0.0, 0.0], t)
        elif self.step == 3:
            # Step 3: Move to right above target
            self.send_goal([0.0, -2.0, 1.0, -3.0, 0.0, 0.0], t)
        elif self.step == 4:
            # Step 4: Drop down to place
            self.send_goal([0.0, -2.0, 1.0, -3.0, 0.0, 0.0], t)
        elif self.step == 5:
            # Step 5: Move up
            self.send_goal([-1.0, -2.0, 1.0, -1.0, 1.0, 0.0], t)
        elif self.step == 6:
            # Step 6: Move back to left
            self.send_goal([1.0, -2.0, 1.0, -1.0, 1.0, 0.0], t)

        self.step = (self.step + 1) % 6
def main(args=None):
    rclpy.init(args=args)
    node = UR10TrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

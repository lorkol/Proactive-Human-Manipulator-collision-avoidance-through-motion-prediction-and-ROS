#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState


class JointStateReader(Node):
    def __init__(self):
        super().__init__('joint_state_reader')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)

    def joint_callback(self, msg):
        joint_order = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        joint_positions = dict(zip(msg.name, msg.position))
        ordered_positions = [joint_positions[joint] for joint in joint_order]
        print(ordered_positions)
        # self.get_logger().info(f"Joint Positions: {ordered_positions}")
        
def main(args=None):
    rclpy.init(args=args)
    node = JointStateReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
from typing import List, Dict, Any

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

pose_seq: NDArray


class JointStateReader(Node):
    def __init__(self):
        super().__init__('joint_state_reader')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)

    def joint_callback(self, msg):
        joint_positions: Dict[str, Any] = dict(zip(msg.name, msg.position))
        self.get_logger().info(f"Joint Positions: {joint_positions}")


class PoseListener(Node):
    def __init__(self):
        super().__init__('pose_listener')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'joint_array',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        global pose_seq
        NUM_FRAMES = 1
        NUM_JOINTS = 15
        DIMENSIONS = 3
        pose_seq = np.array(msg.data).reshape((NUM_FRAMES, NUM_JOINTS, DIMENSIONS))

class UR10TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('ur10_pick_place_loop')
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        self.config: List[float] = []
        self.target_coords: NDArray = np.array([0, 0, 0])
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.step = 0

    def send_trajectory(self, positions, duration_sec):
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
        point.positions = positions
        point.time_from_start = Duration(nanosec=duration_sec)
        traj.points.append(point)
        self.publisher_.publish(traj)
        self.get_logger().info(f'Published trajectory {positions}')

    def timer_callback(self):
        global pose_seq

        pick = np.array([0, -np.pi / 4, np.pi / 2, -np.pi / 2, np.pi / 4, 0])
        place: NDArray = np.array([-1.5, -np.pi / 4, np.pi / 2, -np.pi / 2, np.pi / 4, 0])
        self.target_coords = forward_kinematics(place)
        if len(self.config) == 0:
            self.config = pick.copy()
        goal = place
        print("pose_seq_loop",pose_seq)

        path = a_rrt_star(self.config, goal, pose_seq, self.target_coords, iterations=5)

        print("path", list(path))
        tbsent = path[0]
        if len(path) > 1:
            tbsent = path[1]
        self.send_trajectory(list(tbsent), 10000000)
        self.config = list(tbsent.copy())

################################################
################################################
# Trajectory planning functions


# -------------------- DH & FK --------------------
dh_params = [
    [0,       0,        0.1807,   np.pi/2],
    [0,  -0.4784,       0,        0],
    [0,  -0.36,         0,        0],
    [0,       0,        0.17415,  np.pi/2],
    [0,       0,        0.11985, -np.pi/2],
    [0,       0,        0.11655,  0]
]


def dh_transform(theta: float, a: float, d: float, alpha: float):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                np.cos(alpha),               d],
        [0,              0,                            0,                           1]
    ])


def forward_kinematics(joint_angles: NDArray) -> NDArray:
    T: NDArray = np.eye(4)
    positions: List[NDArray] = []
    for i in range(6):
        theta: float = joint_angles[i] + dh_params[i][0]
        a, d, alpha = dh_params[i][1:]
        A = dh_transform(theta, a, d, alpha)
        T = T @ A
        positions.append(T[:3, 3])
    return np.array(positions)


def get_full_link_points(joint_positions: NDArray, num_points=5) -> NDArray:
    link_points = []
    for i in range(len(joint_positions) - 1):
        start, end = joint_positions[i], joint_positions[i + 1]
        for t in np.linspace(0, 1, num_points):
            interp_pt = start * (1 - t) + end * t
            link_points.append(interp_pt)
    return np.array(link_points)


# -------------------- Potential Field --------------------
def potential(pt, link_params, scale, dth=5) -> float:
    pt = pt * 5
    p1, p2, rad = link_params
    axis = p2 - p1
    cyl_len = np.linalg.norm(axis)
    vec = pt - p1
    axial = np.dot(axis, vec) / cyl_len
    axial1 = np.dot(axis, pt - p2) / cyl_len
    radial = np.sqrt(np.linalg.norm(vec) ** 2 - axial ** 2) if np.linalg.norm(vec) > 0 else 0
    if np.sign(axial) > 0 and abs(axial) < cyl_len and radial < rad:
        return 500 * scale
    elif radial < dth and min(abs(axial), abs(axial1)) < dth:
        return 100 * scale
    return 0.


def extract_links(pose):
    links = []
    tc = (pose[0] + pose[1]) / 2
    bc = (pose[9] + pose[10]) / 2
    rad = max(np.linalg.norm(pose[2] - pose[3]), np.linalg.norm(pose[9] - pose[10])) / 2
    links.append([tc, bc, rad])
    bc = (pose[0] + pose[1]) / 2
    tc = bc + 2 * (pose[1] - pose[8]) * rad / (3 * np.linalg.norm(pose[1] - pose[8]))
    links.append([tc, bc, rad / 3])
    joint_idx_map = [[3, 4, rad / 6], [4, 6, rad / 6], [2, 5, rad / 6], [5, 7, rad / 6],
                     [9, 11, rad / 2], [10, 12, rad / 2], [11, 13, rad / 2], [12, 14, rad / 2]]
    for link in joint_idx_map:
        links.append([pose[link[0]], pose[link[1]], link[2]])
    return links


def compute_APF(pose_seq, coords, target_coords) -> NDArray:
    p_curr = pose_seq[-1]
    p_pred = p_curr + np.random.randn(*p_curr.shape) * 20
    poses: List = [(p_curr / 10) + 100, (p_pred / 10) + 100]
    potentials: NDArray = np.zeros(len(coords))
    for p_no, P in enumerate(poses):
        links = extract_links(P)
        for i, pt in enumerate(coords):
            # repulsive potential
            for link in links:
                potentials[i] += potential(pt, link, 1 / (1 + p_no))

            # attractive potential
            potentials[i] -= 7500 / (np.linalg.norm(target_coords - pt) + 1)
    return potentials


# -------------------- Heuristic & A-RRT* --------------------
def compute_total_apf(link_points, pose_seq, target_coords):
    return np.sum(compute_APF(pose_seq, link_points, target_coords))


def heuristic(qs, qe, pose_seq, target_coords, epsilon=2.0):
    P_s = compute_total_apf(get_full_link_points(forward_kinematics(qs)), pose_seq, target_coords)
    P_e = compute_total_apf(get_full_link_points(forward_kinematics(qe)), pose_seq, target_coords)
    P_max = max(P_s, P_e)
    return (P_e - P_s + 1) / np.exp(epsilon * (1 - P_max / 3000))


def a_rrt_star(start_q, goal_q, pose_seq, target_coords, iterations=100):
    class Node:
        def __init__(self, q):
            self.q = q
            self.parent = None
            self.cost = 0

    def steer(from_q, to_q, step_size=0.2):
        direction = to_q - from_q
        norm = np.linalg.norm(direction)
        return from_q + step_size * direction / norm if norm > 0 else from_q

    def is_safe(q):
        full_link_points = get_full_link_points(forward_kinematics(q))
        return compute_total_apf(full_link_points, pose_seq, target_coords) < 300

    nodes: List[Node] = [Node(start_q)]
    for _ in range(iterations):
        rand_q = np.random.uniform(-np.pi, np.pi, size=6)
        nearest = min(nodes, key=lambda n: heuristic(n.q, rand_q, pose_seq, target_coords))
        new_q = steer(nearest.q, rand_q)
        if is_safe(new_q):
            new_node = Node(new_q)
            new_node.parent = nearest
            new_node.cost = nearest.cost + np.linalg.norm(new_q - nearest.q)
            nodes.append(new_node)
            if np.linalg.norm(new_q - goal_q) < 0.2:
                goal_node = Node(goal_q)
                goal_node.parent = new_node
                nodes.append(goal_node)
                break

    path = []
    current: Node = nodes[-1]
    while current:
        path.append(current.q)
        current = current.parent
    return path[::-1] if len(path) > 1 else [start_q]


def main(args=None):
    rclpy.init(args=args)

    joint_reader = PoseListener()
    trajectory_publisher = UR10TrajectoryPublisher()

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
#!/usr/bin/env python3

from typing import Annotated, Dict, List, TypeAlias, Tuple
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sklearn.neighbors import KDTree
import cupy as cp
from cupy.typing import NDArray
import time
import threading

# Type Aliasing
RobotAnglesVector: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
"""Shape : (6,) representing the 6 joint angles of the robot"""
RobotJointPositions: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (6, 3) representing the x,y,z positions of each of the 6 robot joints'''
HumanPoseSequence: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (N, 15, 3) where N is the number of time steps - Currently 1, each pose has 15 joints with (x,y,z) coordinates'''
HumanPose: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (15, 3) human pose in a single time step each pose has 15 joints with (x,y,z) coordinates'''
Position: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (3,) representing a 3D position vector'''
Link: TypeAlias = Tuple[Position, Position, float]
'''A body link represented by two end positions and a radius'''

# Global vars
destination: RobotAnglesVector = None
'''Current robot destination as read from /joint_destination'''
pose_seq: HumanPoseSequence = None
'''Current human pose sequence as read from /joint_array'''
body_links: List[Link] = []
'''List of human body links extracted from the current human pose'''
robot_joint_angles: RobotAnglesVector = cp.zeros(6)
"""Robot joint positions as in the angle it is currently in as a cupy array of shape (6,)"""

published: RobotAnglesVector = None
apf_th: float = 20.
'''Threshold for the APF value to trigger replanning. Adjust based on environment and robot configuration.'''

dh_params = cp.array([
    [0,       0,        0.1807,   cp.pi/2],
    [0,  -0.4784,       0,        0],
    [0,  -0.36,         0,        0],
    [0,       0,        0.17415,  cp.pi/2],
    [0,       0,        0.11985, -cp.pi/2],
    [0,       0,        0.11655,  0]
])


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
        j_positions: Dict[str, List[float]] = dict(zip(msg.name, msg.position))
        robot_joint_angles = cp.array([j_positions[joint] for joint in joint_order])
        
class DestinationReader(Node):
    """Regarding the Destination ."""
    def __init__(self) -> None:
        super().__init__('destination_reader')
        self.subscription = self.create_subscription(
            JointTrajectoryPoint,
            '/joint_destination',
            self.destination_callback,
            10)

    def destination_callback(self, msg) -> None:
        global destination
        destination = cp.array(msg.positions)

class PoseListener(Node):
    def __init__(self) -> None:
        self.ready: bool = False
        super().__init__('pose_listener')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'joint_array',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg) -> None:
        global pose_seq, body_links
        self.ready: bool = True
        pose_seq = cp.array(msg.data).reshape((1, 15, 3))
        body_links = extract_links_gpu(pose_seq[0])


class UR16TrajectoryPublisher(Node):
    def __init__(self) -> None:
        super().__init__('ur16_trajectory_publisher')
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        thread = threading.Thread(target=self.main_loop, daemon=True)
        thread.start()

    def send_trajectory(self, positions: RobotAnglesVector, duration_nsec: int) -> None:
        global published
        if published is None:
            published = positions
        else:
            published += ((positions - published + cp.pi) % (2 * cp.pi) - cp.pi)
        traj = JointTrajectory()
        traj.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.positions = cp.asnumpy(published).tolist()
        point.time_from_start = Duration(nanosec=duration_nsec)
        traj.points.append(point)
        self.publisher_.publish(traj)

    def main_loop(self) -> None:
        global robot_joint_angles, body_links, published, destination
        time.sleep(0.5)  # Wait for other nodes to initialize
        #Wait until there is a destinaition to go to
        while destination is None and rclpy.ok():
            time.sleep(0.1)

        current: RobotAnglesVector = robot_joint_angles.copy()
        self.send_trajectory(current, 500000000)
        path: List[RobotAnglesVector] = arrt(current, destination, 200)
        print("After initial planning, path length:", len(path))
        step: int = 1
        look_ahead_steps: int = 3 # How many steps ahead to check for APF threshold exceedance
        while rclpy.ok():
            apf: float = APF_gpu(robot_joint_angles, body_links)
            temp: int = step
            
            # Look ahead along the path to see if APF exceeds threshold
            while apf < apf_th and (temp < len(path) and temp - step < look_ahead_steps):
                apf = max(apf, APF_gpu(path[temp], body_links))
                temp += 1
                
            if apf > apf_th:
                self.send_trajectory(robot_joint_angles, 500000000)
                path = arrt(robot_joint_angles, destination, 200)
                print("Replanned path due to apf threshold length:", len(path))
                self.send_trajectory(path[1], 500000000)
                step = 2
                continue

            dist = cp.linalg.norm(((robot_joint_angles - published + cp.pi) % (2 * cp.pi)) - cp.pi)
            if dist < 0.1 and step < len(path):
                next_pos = path[step]
                self.send_trajectory(next_pos, 500000000)
                step += 1

            if step >= len(path) and cp.linalg.norm(((robot_joint_angles - destination + cp.pi) % (2 * cp.pi)) - cp.pi) < 0.1:
                path = arrt(robot_joint_angles, destination, 200)
                print("New phase planned path length:", len(path))
                self.send_trajectory(path[1], 500000000)
                step = 2


# ----------------- GPU-Optimized Utility Functions -----------------

def dh_transform_batch(joints: RobotAnglesVector) -> RobotJointPositions:
    """Returns the x,y,z positions of each joint given the joint angles using DH parameters."""
    T = cp.eye(4)
    positions = []
    for i in range(6):
        theta = joints[i]
        a = dh_params[i][1]
        d = dh_params[i][2]
        alpha = dh_params[i][3]

        cos_theta: float = cp.cos(theta)
        sin_theta: float = cp.sin(theta)
        cos_alpha: float = cp.cos(alpha)
        sin_alpha: float = cp.sin(alpha)

        T_i_minus1_i = cp.array([
            [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta],
            [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [cp.array(0.0), sin_alpha,              cos_alpha,              cp.array(d)],
            [cp.array(0.0), cp.array(0.0),          cp.array(0.0),          cp.array(1.0)]
        ])
        T = T @ T_i_minus1_i
        positions.append(T[:3, 3])  # Extract current joint position

    return cp.stack(positions) * 1000  # shape: (6, 3) in mm

# TODO understand what this returns, also because this reaches capsule_contrib_batch which has unclarity in types
def get_full_link_points_gpu(joints: RobotJointPositions, n: int = 5) :
    start = joints[:-1]  # (5, 3)
    end = joints[1:]     # (5, 3)
    interp = cp.linspace(0, 1, n).reshape(1, n, 1)  # (1, N, 1)
    # Compute interpolated points for each segment
    pts = start[:, None, :] * (1 - interp) + end[:, None, :] * interp  # (5, N, 3)
    return pts.reshape(-1, 3) 

def extract_links_gpu(pose: HumanPose) -> List[Link]:
    links: List[Link] = []

    # torso
    tc: Position = (pose[0] + pose[1]) / 2
    '''Average of left and right shoulders'''
    bc: Position = (pose[9] + pose[10]) / 2
    '''Average of left and right hips'''
    rad: float = cp.maximum(cp.linalg.norm(pose[2] - pose[3]), cp.linalg.norm(pose[9] - pose[10])) / 2
    links.append((tc, bc, rad))

    # head
    bc: Position = (pose[0] + pose[1]) / 2
    direction = pose[1] - pose[8]
    direction_norm: float = cp.linalg.norm(direction)
    if direction_norm < 1e-6:
        direction = cp.zeros_like(direction)
    else:
        direction = direction * rad / (3 * direction_norm)
    tc = bc + 2 * direction
    links.append((tc, bc, rad / 3))

    # other links (arms, legs, etc.)
    joint_idx_map = [
        [3, 4, rad / 6], [4, 6, rad / 6], [2, 5, rad / 6], [5, 7, rad / 6],
        [9, 11, rad / 2], [10, 12, rad / 2], [11, 13, rad / 2], [12, 14, rad / 2]
    ]
    for link in joint_idx_map:
        links.append((pose[link[0]], pose[link[1]], link[2]))

    return links

def capsule_contrib_batch(points: RobotJointPositions, links: List[Link], dth=500) -> float:
    total: float = 0.
    for p1, p2, r in links:
        d_vec = p2 - p1
        d_norm: float = cp.linalg.norm(d_vec)
        if d_norm<1e-6:
            continue
        v = points - p1 # TODO the types mismatch. points is 6x3, but p1 is 3x1
        axial: float = cp.dot(v, d_vec) / d_norm
        proj = cp.outer(axial / d_norm, d_vec)
        radial: float = cp.linalg.norm(v - proj, axis=1)
        d = cp.where(axial < 0, cp.linalg.norm(v, axis=1) - r,
            cp.where(axial > d_norm, cp.linalg.norm(points - p2, axis=1) - r, radial - r))
        contrib = cp.where(d < 0, 2, cp.where(d > dth, 0, cp.cos((d * cp.pi) / (2 * dth))))
        total += contrib.sum()
    return total

def APF_gpu(q: RobotAnglesVector, links: List[Link]) -> float:
    t1 = time.time()
    pts: RobotJointPositions = get_full_link_points_gpu(dh_transform_batch(q))
    ccb: float = capsule_contrib_batch(pts, links)
    print(f"APF computation time: {time.time()-t1:.6f} seconds")
    return ccb

# -------------------------RRT Implementation-------------------------
class RRTNode:
    def __init__(self, q: RobotAnglesVector) -> None:
        self.q: RobotAnglesVector = q
        self.parent: 'RRTNode' = None
        self.cost: float = 0.

def steer(q1: RobotAnglesVector, q2: RobotAnglesVector, step: float = 0.2) -> RobotAnglesVector:
    d = (q2 - q1 + cp.pi) % (2 * cp.pi) - cp.pi
    norm = cp.linalg.norm(d)
    return q1 + d * (step / norm) if norm > 1e-6 else q1

# ----------------- A-RRT* Planning Function -----------------
def arrt(q_start: RobotAnglesVector, q_goal: RobotAnglesVector, n_nodes: int = 100) -> List[RobotAnglesVector]:
    start_t = time.time()
    n_explored: int = 0
    n_used: int = 0

    start_tree: List[RRTNode] = [RRTNode(q_start)]
    goal_tree: List[RRTNode] = [RRTNode(q_goal)]
    itr: int = 0
    while itr < n_nodes:
        q_rand: RobotAnglesVector = q_goal if cp.random.rand() < 0.1 else (cp.random.normal(loc=q_goal, scale=1))
        q_rand = (q_rand + cp.pi) % (2 * cp.pi) - cp.pi
        n_explored += 1
        closest: RRTNode = min(start_tree, key=lambda n: cp.linalg.norm((q_rand - n.q + cp.pi) % (2 * cp.pi) - cp.pi))
        q_new: RobotAnglesVector = steer(closest.q, q_rand)
        
        if APF_gpu(q_new, body_links) > apf_th:
            continue

        itr += 1
        n_used += 1
        node: RRTNode = RRTNode(q_new)
        node.parent = closest
        start_tree.append(node)
        
        #connect step
        connected: bool = False
        closest_goal: RRTNode = min(goal_tree, key=lambda n: cp.linalg.norm((q_new - n.q + cp.pi) % (2 * cp.pi) - cp.pi))
        dir = (q_new - closest_goal.q+ cp.pi) % (2 * cp.pi) - cp.pi
        norm: float = cp.linalg.norm(dir)
        if norm > 0.2:
            dir = dir * (0.2 / norm)  
        else:
            connected = True
        q_added: RobotAnglesVector = closest_goal.q + dir
        par: RRTNode = closest_goal
        while APF_gpu(q_added, body_links) < apf_th and connected==False:
            nn: RRTNode = RRTNode(q_added)
            nn.parent = par
            goal_tree.append(nn)
            par = nn
            q_added = (q_added+dir + cp.pi) % (2 * cp.pi) - cp.pi
            if cp.linalg.norm((q_added - q_new+ cp.pi) % (2 * cp.pi) - cp.pi)<0.2:
                connected = True
                break
        
        if connected == True:
            break
        
        if cp.linalg.norm((q_new - q_goal + cp.pi) % (2 * cp.pi) - cp.pi) < 0.2:
            final: RRTNode = RRTNode(q_goal)
            final.parent = node
            start_tree.append(final)
            break
    
    end_t = time.time()
    print("Planning time:",end_t-start_t)
    print("n_explored",n_explored)
    print("n_used",n_used)
    print("start_tree",len(start_tree))
    print("goal_tree",len(goal_tree))
    path: List[RobotAnglesVector] = []
    node: RRTNode = start_tree[-1]
    while node:
        path.append(node.q)
        node = node.parent
    path = path[::-1]
    
    node = goal_tree[-1]
    while node:
        path.append(node.q)
        node = node.parent
    return path


def main(args=None):
    rclpy.init(args=args)
    pose_listener = PoseListener()
    joint_reader = JointStateReader()
    destination_reader = DestinationReader()
    trajectory_publisher = UR16TrajectoryPublisher()

    executor = MultiThreadedExecutor()
    while not pose_listener.ready and rclpy.ok():
        rclpy.spin_once(pose_listener, timeout_sec=0.1)

    executor.add_node(pose_listener)
    executor.add_node(joint_reader)
    executor.add_node(destination_reader)
    executor.add_node(trajectory_publisher)

    try:
        executor.spin()
    finally:
        joint_reader.destroy_node()
        destination_reader.destroy_node()
        pose_listener.destroy_node()
        trajectory_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

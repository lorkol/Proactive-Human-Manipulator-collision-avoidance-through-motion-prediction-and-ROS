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

# Joint limits for UR16e from URDF configuration
JOINT_LIMITS = {
    'shoulder_pan_joint': {
        'min_position': cp.radians(-360),
        'max_position': cp.radians(360),
        'max_velocity': cp.radians(120),
        'max_acceleration': cp.radians(150),  # Conservative estimate: 150 deg/s²
        'max_deceleration': cp.radians(180)  # Conservative estimate: 180 deg/s²
    },
    'shoulder_lift_joint': {
        'min_position': cp.radians(-360),
        'max_position': cp.radians(360),
        'max_velocity': cp.radians(120),
        'max_acceleration': cp.radians(150),  # Conservative estimate: 150 deg/s²
        'max_deceleration': cp.radians(180)  # Conservative estimate: 180 deg/s²
    },
    'elbow_joint': {
        'min_position': cp.radians(-180),
        'max_position': cp.radians(180),
        'max_velocity': cp.radians(180),
        'max_acceleration': cp.radians(150),  # Conservative estimate: 150 deg/s²
        'max_deceleration': cp.radians(180)  # Conservative estimate: 180 deg/s²
    },
    'wrist_1_joint': {
        'min_position': cp.radians(-360),
        'max_position': cp.radians(360),
        'max_velocity': cp.radians(180),
        'max_acceleration': cp.radians(200),  # Conservative estimate: 200 deg/s²
        'max_deceleration': cp.radians(220)  # Conservative estimate: 220 deg/s²
    },
    'wrist_2_joint': {
        'min_position': cp.radians(-360),
        'max_position': cp.radians(360),
        'max_velocity': cp.radians(180),
        'max_acceleration': cp.radians(200),  # Conservative estimate: 200 deg/s²
        'max_deceleration': cp.radians(220)  # Conservative estimate: 220 deg/s²
    },
    'wrist_3_joint': {
        'min_position': cp.radians(-360),
        'max_position': cp.radians(360),
        'max_velocity': cp.radians(180),
        'max_acceleration': cp.radians(200),  # Conservative estimate: 200 deg/s²
        'max_deceleration': cp.radians(220)  # Conservative estimate: 220 deg/s²
    }
}

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
destination_outdated: bool = False
'''Flag indicating if the destination has been updated'''
pose_seq: HumanPoseSequence = None
'''Current human pose sequence as read from /joint_array'''
body_links: List[Link] = []
'''List of human body links extracted from the current human pose'''
robot_joint_angles: RobotAnglesVector = cp.zeros(6)
"""Robot joint positions as in the angle it is currently in as a cupy array of shape (6,)"""

published: RobotAnglesVector = None
base_APF_Threshold: float = 20.0
'''Threshold for the APF value to trigger replanning. Adjust based on environment and robot configuration.'''
apf_th: float = base_APF_Threshold
'''Adaptive threshold for APF based on distance between start and goal'''
apf_affecting_distance: float = 1.0
'''Distance beyond which APF threshold changes'''

prediction_timesteps: int = 1
'''Number of timesteps to predict into the future for APF checking'''

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
            '/goal_pose',
            self.destination_callback,
            10)

    def destination_callback(self, msg) -> None:
        global destination, destination_outdated
        self.get_logger().info(f"got new goal: {msg.positions}")
        destination = cp.array(msg.positions)
        destination_outdated = True

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
        pose_seq = cp.array(msg.data).reshape((prediction_timesteps, 15, 3)) # TODO adapt to multiple timesteps
        body_links = extract_links_gpu(pose_seq[0])

def wrap_angles(angles: RobotAnglesVector) -> RobotAnglesVector:
    """Wraps angles to the range [-pi, pi]."""
    return (angles + cp.pi) % (2 * cp.pi) - cp.pi

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

    def send_trajectory(self, positions: RobotAnglesVector) -> None:
        global published
        duration_nsec = 500000000 # Default to 0.5 seconds
        if published is None:
            published = positions
        else:
            published += wrap_angles(positions - published)
        traj = JointTrajectory()
        traj.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.positions = cp.asnumpy(published).tolist()
        point.time_from_start = Duration(nanosec=duration_nsec)
        traj.points.append(point)
        self.publisher_.publish(traj)
        self.get_logger().info(f"Published trajectory to: {point.positions}")

    def main_loop(self) -> None:
        global robot_joint_angles, body_links, published, destination, destination_outdated
        time.sleep(0.5)  # Wait for other nodes to initialize
        #Wait until there is a destinaition to go to
        while destination is None:
            # self.get_logger().warning(str(destination))
            time.sleep(0.2)

        current: RobotAnglesVector = robot_joint_angles.copy()
        self.send_trajectory(current) # Send current position to initialize
        path: List[RobotAnglesVector] = arrt(current, destination, 200) # Initial planning
        self.get_logger().info(f"After initial planning, path length: {len(path)}")
        # Before sending first motion step, verify the segment from current to path[1] is safe
        if len(path) > 1:
            if not is_segment_safe(current, path[1], steps=12):
                self.get_logger().warning("Initial step would interpolate through a human link — stopping and replanning.")
                # do not send path[1], keep stopped and replan on next loop
            else:
                self.send_trajectory(path[1])
                step = 2
        step: int = 1
        '''Current step along the planned path'''
        look_ahead_steps: int = 3
        '''How many steps ahead to check for APF threshold exceedance'''
        while rclpy.ok():
            apf: float = APF_gpu(robot_joint_angles, body_links)
            temp: int = step  # Variable to look ahead along the path without modifying step

            # Check whether destination has been updated
            if destination_outdated:
                self.get_logger().info("Destination updated, replanning path.")
                path = arrt(robot_joint_angles, destination, 200) # Replan path
                self.get_logger().info(f"Replanned path length: {len(path)}")
                if len(path) > 1 and is_segment_safe(robot_joint_angles, path[1], steps=12):
                    self.send_trajectory(path[1]) # Send first step in new path
                    step = 2 # Reset step to 2 since we just sent path[1]
                else:
                    self.get_logger().warning("First step of replanned path unsafe — staying stopped and will attempt replanning next loop.")
                destination_outdated = False
                continue

            # Look ahead along the path to see if APF exceeds threshold
            while apf < apf_th and (temp < len(path) and temp - step < look_ahead_steps):
                apf = max(apf, APF_gpu(path[temp], body_links))
                temp += 1

            # Replan if APF threshold exceeded
            if apf > apf_th:
                self.get_logger().warning(f"APF threshold exceeded: apf={apf}, apf_th={float(apf_th)}")
                self.send_trajectory(robot_joint_angles) # Stop movement
                path = arrt(robot_joint_angles, destination, 200) # Replan path
                self.get_logger().warning(f"Replanned path due to apf threshold length: {len(path)}")
                if len(path) > 1 and is_segment_safe(robot_joint_angles, path[1], steps=12):
                    self.send_trajectory(path[1]) # Send first step in new path
                    step = 2 # Reset step to 2 since we just sent path[1]
                else:
                    self.get_logger().warning("First step of replanned path unsafe — staying stopped and will attempt replanning next loop.")
                continue

            dist_from_published = cp.linalg.norm(((robot_joint_angles - published + cp.pi) % (2 * cp.pi)) - cp.pi)
            # Move to next point if close enough to last published point
            if dist_from_published < 0.1 and step < len(path):
                next_pos = path[step] # Get next position
                # Check interpolated segment safety before sending
                if is_segment_safe(robot_joint_angles, next_pos, steps=8):
                    self.send_trajectory(next_pos) # Send next position
                    step += 1 # Advance to next step
                else:
                    self.get_logger().warning("Next segment unsafe, stopping and replanning.")
                    # Stop and trigger replanning on next loop
                    self.send_trajectory(robot_joint_angles)
                    path = arrt(robot_joint_angles, destination, 200)
                    self.get_logger().info(f"Replanned path length: {len(path)}")
            
            # Check if destination reached
            dist = cp.linalg.norm(((robot_joint_angles - destination + cp.pi) % (2 * cp.pi)) - cp.pi)
            if step >= len(path) and dist < 0.1:
                time.sleep(0.2)  # Idle if at destination 
            
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

    # Return positions in meters to match human pose units (pose_seq is in meters).
    return cp.stack(positions)  # shape: (6, 3) in meters

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

def capsule_contrib_batch(points: RobotJointPositions, links: List[Link], dth=0.5) -> float:
    """Calculate the artificial potential field contribution from all capsules (human body links).
    
    Units: everything is in meters. `points` are robot link points in meters and `links`
    are human capsule definitions in meters. `dth` is in meters (default 0.5 m).
    Args:
        points: Shape (N, 3) array of points along robot links
        links: List of (start_point, end_point, radius) tuples representing human body segments
        dth: Distance threshold beyond which potential field becomes zero (default: 0.5 m)
    """
    total: float = 0.
    for p1, p2, r in links:
        d_vec = p2 - p1
        d_norm: float = cp.linalg.norm(d_vec)
        if d_norm < 1e-6:
            continue
            
        # Vectors from capsule start to each robot point
        v = points - p1
        
        # Project these vectors onto capsule direction
        axial: float = cp.dot(v, d_vec) / d_norm  # Projection lengths
        proj = cp.outer(axial / d_norm, d_vec)    # Projected points
        
        # Calculate perpendicular distance from points to capsule line
        radial: float = cp.linalg.norm(v - proj, axis=1)
        
        # Distance to capsule surface:
        # - If point is before start of capsule: distance to start minus radius
        # - If point is after end of capsule: distance to end minus radius
        # - Otherwise: perpendicular distance to line minus radius
        d = cp.where(axial < 0, 
                    cp.linalg.norm(v, axis=1) - r,  # Distance to start point
                    cp.where(axial > d_norm, 
                            cp.linalg.norm(points - p2, axis=1) - r,  # Distance to end point
                            radial - r))  # Distance to cylinder
        
        # If any point is inside or exactly on the capsule surface (d <= 0) it's a hard collision: reject.
        if cp.any(d <= 0):
            return float('inf')  # Immediately reject configurations in collision

        # For non-colliding configurations, produce a strong repulsive potential in meters.
        # Use safe minimum distance to avoid div/overflow and choose a decay tuned for meters.
        safe_d = cp.maximum(d, 0.01)  # 1 cm lower bound to avoid numerical issues
        # Exponential repulsion: large magnitude near the body, quickly decays with distance.
        contrib = cp.where(d > dth,
                         0.0,  # Far from capsule
                         80.0 * cp.exp(-safe_d / 0.1))  # Decay length = 0.1 m (10 cm)

        total += contrib.sum()
    return total

def APF_gpu(q: RobotAnglesVector, links: List[Link]) -> float:
    pts: RobotJointPositions = get_full_link_points_gpu(dh_transform_batch(q))
    return capsule_contrib_batch(pts, links)


def is_segment_safe(q_from: RobotAnglesVector, q_to: RobotAnglesVector, steps: int = 20) -> bool:
    """Sample the joint-space segment between q_from and q_to and ensure no sample exceeds APF threshold.

    Uses linear interpolation in joint space with angle wrapping.
    Returns True if all samples have APF <= apf_th and are collision-free.
    """
    global body_links, apf_th
    for alpha in cp.linspace(0.0, 1.0, steps):
        q_sample = wrap_angles((1 - alpha) * q_from + alpha * q_to)
        apf_val = APF_gpu(q_sample, body_links)
        if apf_val > apf_th:
            return False
    return True

# -------------------------RRT Implementation-------------------------
def is_within_joint_limits(q: RobotAnglesVector) -> bool:
    """Check if a configuration is within the robot's joint limits."""
    joint_names = list(JOINT_LIMITS.keys())
    for i, joint_name in enumerate(joint_names):
        if q[i] < JOINT_LIMITS[joint_name]['min_position'] or \
           q[i] > JOINT_LIMITS[joint_name]['max_position']:
            return False
    return True

class RRTNode:
    def __init__(self, q: RobotAnglesVector) -> None:
        self.q: RobotAnglesVector = q
        self.parent: 'RRTNode' = None
        self.cost: float = 0.

def steer(q1: RobotAnglesVector, q2: RobotAnglesVector, step: float = 0.2) -> RobotAnglesVector:
    d = wrap_angles(q2 - q1)
    norm = cp.linalg.norm(d)
    return q1 + d * (step / norm) if norm > 1e-6 else q1

# ----------------- A-RRT* Planning Function -----------------
def arrt(q_start: RobotAnglesVector, q_goal: RobotAnglesVector, n_nodes: int = 100) -> List[RobotAnglesVector]:
    start_t = time.time()
    n_explored: int = 0
    n_used: int = 0

    start_tree: List[RRTNode] = [RRTNode(q_start)]
    goal_tree: List[RRTNode] = [RRTNode(q_goal)]
    
    # Adaptive APF threshold based on start-goal distance
    global apf_th
    apf_th = base_APF_Threshold
    # Avoid division by zero and only scale threshold when apf_affecting_distance is positive
    dist_norm = float(cp.linalg.norm(wrap_angles(q_start - q_goal)))
    if apf_affecting_distance > 1e-6 and dist_norm > apf_affecting_distance:
        apf_th = base_APF_Threshold * (dist_norm / apf_affecting_distance)
    itr: int = 0
    while itr < n_nodes:
        q_rand: RobotAnglesVector = q_goal if cp.random.rand() < 0.1 else (cp.random.normal(loc=q_goal, scale=1)) # Randomly sample around goal 10% of the time
        q_rand = wrap_angles(q_rand)
        n_explored += 1
        closest: RRTNode = min(start_tree, key=lambda n: cp.linalg.norm(wrap_angles(q_rand - n.q)))
        q_new: RobotAnglesVector = steer(closest.q, q_rand) # New node towards random sample
        
        if APF_gpu(q_new, body_links) > apf_th:
            continue

        itr += 1
        n_used += 1
        node: RRTNode = RRTNode(q_new)
        node.parent = closest
        start_tree.append(node)
        
        #connect step
        connected: bool = False
        closest_goal: RRTNode = min(goal_tree, key=lambda n: cp.linalg.norm(wrap_angles(q_new - n.q))) # Find closest node in goal tree
        dir = wrap_angles(q_new - closest_goal.q) # Direction from closest goal to new node
        norm: float = cp.linalg.norm(dir)
        if norm > 0.2:
            dir = dir * (0.2 / norm)  
        else:
            connected = True
        q_added: RobotAnglesVector = closest_goal.q + dir
        par: RRTNode = closest_goal
        
        # Extend goal tree towards new node
        while APF_gpu(q_added, body_links) < apf_th and connected==False:
            # self.get_logger().info("adding nodes to goal tree")
            nn: RRTNode = RRTNode(q_added)
            nn.parent = par
            goal_tree.append(nn)
            par = nn
            q_added = wrap_angles(q_added + dir)
            if cp.linalg.norm(wrap_angles(q_added - q_new)) < 0.2:
                connected = True
                break
        
        if connected == True:
            break
        
        # Check if new node is close enough to goal
        if cp.linalg.norm(wrap_angles(q_new - q_goal)) < 0.2:
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

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sklearn.neighbors import KDTree
import cupy as cp
import heapq
import time
import threading

# Global vars
pose_seq, joint_positions, body_links = None, None, []
published = None
apf_th=20

dh_params = cp.array([
    [0,       0,        0.1807,   cp.pi/2],
    [0,  -0.4784,       0,        0],
    [0,  -0.36,         0,        0],
    [0,       0,        0.17415,  cp.pi/2],
    [0,       0,        0.11985, -cp.pi/2],
    [0,       0,        0.11655,  0]
])


class JointStateReader(Node):
    def __init__(self):
        super().__init__('joint_state_reader')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)

    def joint_callback(self, msg):
        global joint_positions
        joint_order = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        j_positions = dict(zip(msg.name, msg.position))
        joint_positions = cp.array([j_positions[joint] for joint in joint_order])


class PoseListener(Node):
    def __init__(self):
        self.ready = False
        super().__init__('pose_listener')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'joint_array',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        global pose_seq, body_links
        self.ready = True
        pose_seq = cp.array(msg.data).reshape((1, 15, 3))
        body_links = extract_links_gpu(pose_seq[0])


class UR16TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('ur16_pick_place_loop')
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        thread = threading.Thread(target=self.main_loop, daemon=True)
        thread.start()

    def send_trajectory(self, positions, duration_nsec):
        global published
        if published is None:
            published = positions
        else:
            published+=((positions-published+cp.pi)%(2*cp.pi)-cp.pi)
        traj = JointTrajectory()
        traj.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.positions = cp.asnumpy(published).tolist()
        point.time_from_start = Duration(nanosec=duration_nsec)
        traj.points.append(point)
        self.publisher_.publish(traj)

    def main_loop(self):
        global joint_positions, body_links, published
        time.sleep(0.5)
        pick = cp.array([0.0, -1.0, 1.5, -3.0, 0.0, 0.0])
        place = cp.array([-2.0, -1.0,  1.5, -3.0,  0.0, 0.0])
        phase_sequence = [(pick.copy(), place.copy()), (place.copy(), pick.copy())]
        phase = 0

        current = pick
        self.send_trajectory(current, 500000000)
        path = arrt(current, phase_sequence[phase][1], 200)
        step = 1
        # published = current

        while rclpy.ok():
            apf = APF_gpu(joint_positions, body_links)
            temp = step
            while apf<apf_th and (temp<len(path) and temp-step<3):
                apf = max(apf,APF_gpu(path[temp], body_links))
                temp+=1
                
            # print("apf",apf)
            # if APF_gpu(joint_positions, body_links) > 10:
            if apf>apf_th:
                self.send_trajectory(joint_positions, 500000000)
                path = arrt(joint_positions, phase_sequence[phase][1], 200)
                self.send_trajectory(path[1], 500000000)
                step = 2
                # published = path[0]
                continue

            dist = cp.linalg.norm(((joint_positions - published + cp.pi) % (2 * cp.pi)) - cp.pi)
            # print(dist)
            # print("phase",phase)
            if dist < 0.1 and step < len(path):
                next_pos = path[step]
                self.send_trajectory(next_pos, 500000000)
                # published = next_pos
                step += 1

            if step >= len(path) and cp.linalg.norm(((joint_positions - phase_sequence[phase][1] + cp.pi) % (2 * cp.pi)) - cp.pi)<0.1:
                phase = (phase + 1) % 2
                # print('new_phase',phase)
                # print('new_target',phase_sequence[phase][1])
                # print('new_path',path)
                path = arrt(joint_positions, phase_sequence[phase][1], 200)
                self.send_trajectory(path[1], 500000000)
                step = 2
                # published = path[0]


# ----------------- GPU-Optimized Utility Functions -----------------

def dh_transform_batch(joints):
    T = cp.eye(4)
    positions = []
    for i in range(6):
        theta = joints[i]
        a = dh_params[i][1]
        d = dh_params[i][2]
        alpha = dh_params[i][3]

        cos_theta = cp.cos(theta)
        sin_theta = cp.sin(theta)
        cos_alpha = cp.cos(alpha)
        sin_alpha = cp.sin(alpha)

        A = cp.array([
            [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta],
            [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [cp.array(0.0), sin_alpha,              cos_alpha,              cp.array(d)],
            [cp.array(0.0), cp.array(0.0),          cp.array(0.0),          cp.array(1.0)]
        ])
        T = T @ A
        positions.append(T[:3, 3])  # Extract current joint position

    return cp.stack(positions) * 1000  # shape: (6, 3) in mm



def get_full_link_points_gpu(joints, n=5):
    # print(joints.shape)
    start = joints[:-1]  # (5, 3)
    end = joints[1:]     # (5, 3)
    interp = cp.linspace(0, 1, n).reshape(1, n, 1)  # (1, N, 1)
    # Compute interpolated points for each segment
    pts = start[:, None, :] * (1 - interp) + end[:, None, :] * interp  # (5, N, 3)
    return pts.reshape(-1, 3) 

def extract_links_gpu(pose):
    links = []

    # torso
    tc = (pose[0] + pose[1]) / 2
    bc = (pose[9] + pose[10]) / 2
    rad = cp.maximum(cp.linalg.norm(pose[2] - pose[3]), cp.linalg.norm(pose[9] - pose[10])) / 2
    links.append((tc, bc, rad))

    # head
    bc = (pose[0] + pose[1]) / 2
    direction = pose[1] - pose[8]
    direction_norm = cp.linalg.norm(direction)
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

def capsule_contrib_batch(points, links, dth=500):
    total = 0
    for p1, p2, r in links:
        d_vec = p2 - p1
        d_norm = cp.linalg.norm(d_vec)
        if d_norm<1e-6:
            continue
        v = points - p1
        axial = cp.dot(v, d_vec) / d_norm
        proj = cp.outer(axial / d_norm, d_vec)
        radial = cp.linalg.norm(v - proj, axis=1)
        d = cp.where(axial < 0, cp.linalg.norm(v, axis=1) - r,
            cp.where(axial > d_norm, cp.linalg.norm(points - p2, axis=1) - r, radial - r))
        contrib = cp.where(d < 0, 2, cp.where(d > dth, 0, cp.cos((d * cp.pi) / (2 * dth))))
        total += contrib.sum()
    return total

def APF_gpu(q, links):
    pts = get_full_link_points_gpu(dh_transform_batch(q))
    return capsule_contrib_batch(pts, links)


# ----------------- A-RRT* Planning Function -----------------
def arrt(q_start, q_goal, n_nodes=100):
    start_t = time.time()
    n_explored=0
    n_used=0
    class Node:
        def __init__(self, q):
            self.q = q
            self.parent = None
            self.cost = 0

    def steer(q1, q2, step=0.2):
        d = (q2 - q1 + cp.pi) % (2 * cp.pi) - cp.pi
        norm = cp.linalg.norm(d)
        return q1 + d * (step / norm) if norm > 1e-6 else q1
        

    start_tree = [Node(q_start)]
    goal_tree = [Node(q_goal)]
    itr=0
    while itr<n_nodes:
        q_rand = q_goal if cp.random.rand() < 0.1 else (cp.random.normal(loc=q_goal, scale=1))
        q_rand = (q_rand + cp.pi) % (2 * cp.pi) - cp.pi
        n_explored+=1
        closest = min(start_tree, key=lambda n: cp.linalg.norm((q_rand - n.q + cp.pi) % (2 * cp.pi) - cp.pi))
        q_new = steer(closest.q, q_rand)
        

        if APF_gpu(q_new, body_links) > apf_th:
            continue
        
        itr+=1
        n_used+=1
        node = Node(q_new)
        node.parent = closest
        start_tree.append(node)
        
        #connect step
        connected = False
        closest_goal = min(goal_tree, key=lambda n: cp.linalg.norm((q_new - n.q + cp.pi) % (2 * cp.pi) - cp.pi))
        dir = (q_new - closest_goal.q+ cp.pi) % (2 * cp.pi) - cp.pi
        norm = cp.linalg.norm(dir)
        if norm > 0.2:
            dir = dir * (0.2 / norm)  
        else:
            connected=True
        q_added = closest_goal.q+dir
        par = closest_goal
        while APF_gpu(q_added, body_links)<apf_th and connected==False:
            nn = Node(q_added)
            nn.parent = par
            goal_tree.append(nn)
            par = nn
            q_added = (q_added+dir + cp.pi) % (2 * cp.pi) - cp.pi
            if cp.linalg.norm((q_added - q_new+ cp.pi) % (2 * cp.pi) - cp.pi)<0.2:
                connected=True
                break
        
        if connected == True:
            break
        
        if cp.linalg.norm((q_new - q_goal + cp.pi) % (2 * cp.pi) - cp.pi) < 0.2:
            final = Node(q_goal)
            final.parent = node
            start_tree.append(final)
            break
    
    end_t = time.time()
    print("Planning time:",end_t-start_t)
    print("n_explored",n_explored)
    print("n_used",n_used)
    print("start_tree",len(start_tree))
    print("goal_tree",len(goal_tree))
    path = []
    node = start_tree[-1]
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
    trajectory_publisher = UR16TrajectoryPublisher()

    executor = MultiThreadedExecutor()
    while not pose_listener.ready and rclpy.ok():
        rclpy.spin_once(pose_listener, timeout_sec=0.1)

    executor.add_node(pose_listener)
    executor.add_node(joint_reader)
    executor.add_node(trajectory_publisher)

    try:
        executor.spin()
    finally:
        joint_reader.destroy_node()
        pose_listener.destroy_node()
        trajectory_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

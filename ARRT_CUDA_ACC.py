#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sklearn.neighbors import KDTree
import numpy as np
import heapq
import queue
import time
import threading
import cupy as cp

global pose_seq, joint_positions, body_links
body_links=[]

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
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        j_positions = dict(zip(msg.name, msg.position))
        joint_positions = cp.array([j_positions[joint] for joint in joint_order])
    

# class PoseListener(Node):
    
#     def __init__(self):
#         self.ready = False
#         super().__init__('pose_listener')
#         self.subscription = self.create_subscription(
#             Float32MultiArray,
#             'joint_array',
#             self.listener_callback,
#             10
#         )
#     def listener_callback(self, msg):
#         self.ready = True
#         # self.get_logger().info("Received first pose")
#         global pose_seq,body_links
#         NUM_FRAMES = 1
#         NUM_JOINTS = 15
#         DIMENSIONS = 3
#         pose_seq = cp.array(msg.data).reshape((NUM_FRAMES, NUM_JOINTS, DIMENSIONS))
#         for i,pose in enumerate(pose_seq):
#             body_links = extract_links(pose) 

#     def joint_callback(self, msg):
        # global joint_positions
        # joint_order = [
        #     'shoulder_pan_joint',
        #     'shoulder_lift_joint',
        #     'elbow_joint',
        #     'wrist_1_joint',
        #     'wrist_2_joint',
        #     'wrist_3_joint'
        # ]
        # j_positions = dict(zip(msg.name, msg.position))
        # joint_positions = [j_positions[joint] for joint in joint_order]
    

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
        self.ready = True
        # self.get_logger().info("Received first pose")
        global pose_seq,body_links
        NUM_FRAMES = 1
        NUM_JOINTS = 15
        DIMENSIONS = 3
        pose_seq = np.array(msg.data).reshape((NUM_FRAMES, NUM_JOINTS, DIMENSIONS))
        for i,pose in enumerate(pose_seq):
            body_links = extract_links(cp.array(pose)) 

class UR16TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('ur16_pick_place_loop')
        self.publisher_ = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        
        self.step = 0
        
        # launch the main loop in a separate thread
        self.running = True
        thread = threading.Thread(target=self.main_loop, daemon=True)
        thread.start()
        
    ###########################MAIN##############################    
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
        point.positions = cp.asnumpy(positions).tolist()
        point.time_from_start = Duration(nanosec=duration_sec)
        traj.points.append(point)
        self.publisher_.publish(traj)

    global initial_c, path, phase, step
    initial_c =0
    
    phase, step = 0, 0

    def main_loop(self):
        global pose_seq, initial_c, path,phase, step, joint_positions
        published=cp.array([])
        time.sleep(0.5)
        no_nodes=200
        pick = cp.array([0.0, -1.0, 1.5, -3.0, 0.0, 0.0])
        place = cp.array([-2.0, -1.0,  1.5, -3.0,  0.0, 0.0])
        phase_sequence = [(pick, place), (place, pick)]
        while rclpy.ok() and self.running: 
            if initial_c==0:
                current_config=pick.copy()
                self.send_trajectory(current_config, 500000000)
                published=current_config.copy()
                path = arrt(current_config.copy(), phase_sequence[phase][1].copy(), no_nodes)
                # print(path)
                initial_c+=1
            
            apf = APF(joint_positions)
            # print("APF",apf)
            # print("phase",phase)
            if apf > 10: #stop and replan
                # print("APF",apf)
                self.send_trajectory(list(joint_positions), 500000000)
                # print("Replanning APF ABOVE 10")
                published=published+((joint_positions-published+np.pi)%(2*np.pi)-np.pi)
                new_path = arrt(joint_positions.copy(), phase_sequence[phase][1].copy(),no_nodes)
                # if validate_path(new_path):
                #     path = new_path.copy()
                # else:
                #     print("WARNING: path still dangerous after planning — replanning again...")
                # print("Replanning---------------------",new_path)
                
                if len(new_path)>=1:
                    path = new_path.copy()
                    self.send_trajectory(list(path[1]), 500000000)
                    published=published+((path[1]-published+np.pi)%(2*np.pi)-np.pi)
                    time.sleep(0.5)
                    step=2
                continue
            
            else: #iterate through the path or change phase
                if step>=len(path): #change phase
                    check_p=cp.linalg.norm([((joint_positions.copy()-phase_sequence[phase][1].copy()+np.pi)%(2*np.pi)-np.pi)])
                    # print(check_p,phase)
                    if check_p<0.15:
                        
                        phase=(phase+1)%2
                        # print("phase change",phase)
                    path = arrt(joint_positions.copy(), phase_sequence[phase][1].copy(),no_nodes)
                    # print(path)
                    step=1
                check=cp.linalg.norm(((joint_positions.copy()-published+np.pi)%(2*np.pi)-np.pi))
                # print("Check value and phase",check, phase)
                #publish next step if past published config has been reached
                if check<0.15:
                    # print("step increments")
                    tbs = path[step].copy()
                    tbs = published+((tbs-published+np.pi)%(2*np.pi)-np.pi)
                    self.send_trajectory(list(tbs), 500000000)
                    # joint_array=normalize_joints(path[step])
                    # print("joint",joint_array)
                    # self.send_trajectory(list(joint_array), 500000000)
                    published=path[step].copy()
                    published = tbs.copy()
                    step+=1

def normalize_joints(joints):
    return (cp.array(joints) + np.pi) % (2 * np.pi) - np.pi

def validate_path(path):
    for q in path:
        if APF(q) > 7:
            return False
    return True 

dh_params = [
    [0,       0,        0.1807,   np.pi/2],
    [0,  -0.4784,       0,        0],
    [0,  -0.36,         0,        0],
    [0,       0,        0.17415,  np.pi/2],
    [0,       0,        0.11985, -np.pi/2],
    [0,       0,        0.11655,  0]
]

def dh_transform(theta, a, d, alpha):
    return cp.array([
        [cp.cos(theta).get(), -cp.sin(theta).get()*cp.cos(alpha).get(),  cp.sin(theta).get()*cp.sin(alpha).get(), a*cp.cos(theta).get()],
        [cp.sin(theta).get(),  cp.cos(theta).get()*cp.cos(alpha).get(), -cp.cos(theta).get()*cp.sin(alpha).get(), a*cp.sin(theta).get()],
        [0,              cp.sin(alpha).get(),                cp.cos(alpha).get(),               d],
        [0,              0,                            0,                           1]
    ])   

def forward_kinematics(joint_angles):
    T = cp.eye(4)
    positions = []
    for i in range(6):
        theta = joint_angles[i] + dh_params[i][0]
        a, d, alpha = dh_params[i][1:]
        A = dh_transform(theta, a, d, alpha)
        T = T @ A
        positions.append(T[:3, 3])
    return 1000*cp.array(positions)

def get_full_link_points(joint_positions, num_points=5):
    link_points = []
    for i in range(len(joint_positions) - 1):
        start, end = joint_positions[i], joint_positions[i + 1]
        for t in cp.linspace(0, 1, num_points):
            interp_pt = start * (1 - t) + end * t
            link_points.append(interp_pt)
    return cp.array(link_points)

def capsule_contrib(pt,params,dth = 500):
    p1, p2, r = params
    p1=cp.array(p1)
    p2=cp.array(p2)
    if cp.linalg.norm(p2-p1)<1e-6:
        return 0
    d_axial = cp.dot(pt-p1,p2-p1)/cp.linalg.norm(p2-p1)
    d_radial = cp.sqrt(cp.linalg.norm(pt-p1)**2 - d_axial**2)
    if d_axial<0:
        d = cp.linalg.norm(pt-p1)-r  
        
    elif d_axial>cp.linalg.norm(p2-p1):
        d = cp.linalg.norm(pt-p2)-r  
        
    else:
        d = d_radial-r
    
    if d<0:
        return 2
    elif d>dth:
        return 0
    else:
        return cp.cos((d*np.pi)/(2*dth)).get()

def extract_links(pose):
    links=[]
    
    #torso
    tc = (pose[0]+pose[1])/2
    bc = (pose[9]+pose[10])/2
    rad = max(cp.linalg.norm(pose[2]-pose[3]),cp.linalg.norm(pose[9]-pose[10]))/2
    # print(tc,bc,rad)
    links.append([tc,bc,rad])

    #head
    bc = (pose[0]+pose[1])/2
    tc = bc + 2*(pose[1]-pose[8])*rad/(3*cp.linalg.norm(pose[1]-pose[8]))
    links.append([tc,bc,rad/3])
    # print()
    #corres joints for remaining links
    joint_idx_map = [[3,4,rad/6],[4,6,rad/6],[2,5,rad/6],[5,7,rad/6],[9,11,rad/2],[10,12,rad/2],[11,13,rad/2],[12,14,rad/2]]
    for link in joint_idx_map:
        links.append([pose[link[0]],pose[link[1]],link[2]])
    # print("links",links)
    return links

def APF(q):
    mani_pts = get_full_link_points(forward_kinematics(q))
    
    global pose_seq,body_links #body_links: 10x(point, point, scalar), mani_pts: 25xpoint
    # P = 0
    # wt = [1] * len(pose_seq)
    
    # for pt in mani_pts:
    #     P += sum(capsule_contrib(pt, l) for l in body_links)
    P = cp.sum(cp.array([cp.sum(cp.array([capsule_contrib(pt,l) for l in body_links])) for pt in mani_pts]))    
    return P

def arrt(q_start,q_goal,n_nodes=100):
    start_t = time.time()
    class Node:
        def __init__(self,q):
            self.q = q
            self.parent = None
            self.sum_f = 0
            self.f=0
    
    def js_dist(q1,q2):
        min_diff = (q2-q1 + np.pi) % (2 * np.pi) - np.pi
        # min_diff = q2-q1
        return cp.linalg.norm(min_diff)

    def steer(q1,q2,step_size = 0.10):
        direction = (q2-q1+np.pi)%(2*np.pi)-np.pi
        if cp.linalg.norm(direction)<1e-6:
            return cp.zeros(q1.shape)
        # direction = q2-q1:
        q_steered = q1+direction*(step_size/cp.linalg.norm(direction))
        # return (q_steered + np.pi) % (2 * np.pi) - np.pi
        return q_steered

    def heu_score(q1,q2,q_goal,e = 1):
        P1 = APF(cp.array(q1))
        P2 = APF(cp.array(q2))
        # u_1 = unit_vect(q1,q2)
        # u_2 = unit_vect(q1,q_goal)
        return ((1+P2-P1)/(cp.exp(e*(1-max(P1,P2)))))
    
    def knn_kdtree(nodes,q,k=5):
        tree = KDTree([cp.asnumpy(n.q) for n in nodes])
        _,idxs = tree.query([cp.asnumpy(q)],min(k,len(nodes)))
        near_nodes = [nodes[i] for i in idxs[0]]
        return near_nodes
    
    nodes= [Node(q_start)]
    itr_n = 0
    print("Planning...")
    n_explored=0
    n_used=0
    while itr_n<n_nodes:
        q_rand = q_goal if cp.random.rand()<0.1 else cp.random.normal(loc=q_goal, scale=0.5, size=6)
        q_rand = (q_rand+np.pi)%(2*np.pi)-np.pi
        n_explored+=1
        i_min = 0
        min_dist = 10000000
        for i in range(len(nodes)):
            if js_dist(q_rand, nodes[i].q) < min_dist:
                i_min = i
                min_dist = js_dist(q_rand,nodes[i].q)
        q_nearest = nodes[i_min].q

        #Deduce the new node's config
        q_new = steer(q_nearest,q_rand)
        # print('q_new',q_new)
        apf = APF(q_new)
        if apf> 8:  
            continue
        n_used+=1
        itr_n+=1
        
        nn = Node(q_new)
        nn.f = heu_score(q_nearest,q_new,q_goal)*0.2 #f = h*g
        nn.sum_f = nodes[i_min].f + nn.f
        Q_near = knn_kdtree(nodes,q_new) # k-parameter
        # print('Q_near',[qn.q for qn in Q_near])
        f_s=[heu_score(q_i.q,q_new,q_goal)*0.2 for q_i in Q_near]
        # print('scores',f_s)
        sumf_s = [Q_near[i].sum_f+f_s[i] for i in range(len(f_s))]
        # print('cumu_scores',sumf_s)
        nn.sum_f = min(sumf_s)
        nn.parent = Q_near[sumf_s.index(min(sumf_s))]
        # print('parent',nn.parent.q)where
        nodes.append(nn)
        
        if js_dist(q_new,q_goal)<0.2:
            fin_n = Node(q_goal)
            fin_n.parent = nn
            nodes.append(fin_n)
            break
        
        p = []
    current  = nodes[-1]
    while current:
        p.append(current.q)
        current = current.parent
    end_t = time.time()
    print("Planning time:",start_t-end_t)
    print("n_explored:",n_explored)
    print("n_used:",n_used)
    # print("PATH", path[::-1])
    return p[::-1]

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




#!/usr/bin/env python3
from typing import Annotated, Dict, List, TypeAlias, Tuple
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import cupy as cp
from cupyx.scipy.spatial import KDTree
from cupy.typing import NDArray
import time
import threading

#---------------------------------------------------------- Type Aliasing
#Holonomic
RobotAnglesVector: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (6,) representing the 6 joint angles of the robot'''
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

# Kinodynamic
RobotVelocitiesVector: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (6,) representing the 6 joint angle velocities of the robot'''
RobotKinodynamicState: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (12,) representing the 6 joint angles and 6 joint velocities of the robot'''

class RobotState:
    """Represents the kinodynamic state of the robot: joint angles and joint velocities."""
    #Static fields (class variables)
    DOF: int = 6
    '''Degrees of freedom'''
    MAX_POSITIONS: RobotAnglesVector = cp.array([6.28, 6.28, 3.14, 6.28, 6.28, 6.28])
    '''Max angular positions in rad according to the official UR16e documentation'''
    MAX_VELOCITIES: RobotVelocitiesVector = cp.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14])
    '''Max angular velocities in rad/s according to the official UR16e documentation'''
    MAX_ACCELERATIONS: RobotAnglesVector = cp.array([6.28, 6.28, 6.28, 6.28, 6.28, 6.28])
    '''Example: max angular accelerations in rad/s^2 according to the official UR16e documentation'''

    def __init__(self, angles: RobotAnglesVector, velocities: RobotVelocitiesVector) -> None:
        self.angles: RobotAnglesVector = angles.copy()
        self.velocities: RobotVelocitiesVector = velocities.copy()

        
    ############Methods to convert to vectors
    def angles_to_vector(self) -> RobotAnglesVector:
        """returns the joint angles as a cupy array of shape (6,)"""
        return self.angles
    
    def velocities_to_vector(self) -> RobotVelocitiesVector:
        """returns the joint velocities as a cupy array of shape (6,)"""
        return self.velocities

    def to_vector(self) -> RobotKinodynamicState:
        """returns the joint angles and velocities concatenated as a single cupy array of shape (12,)"""
        return cp.concatenate((self.angles_to_vector(), self.velocities_to_vector()))


ControlInput: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray] 
'''A control input is represented as a vector of joint angle accelerations'''

#------------------------------------------------------- RRT Functions
class KinodynamicRRTNode:
    """A node in the kinodynamic RRT. Holds the robot state, parent node, and cost to reach this node."""
    def __init__(self, state: RobotState, time_stamp: int, control_input_used: ControlInput = None) -> None:
        self.robot_state: RobotState = state
        '''The robot state at this node'''
        self.parent: 'KinodynamicRRTNode' = None
        '''Parent node in the RRT'''
        self.cost: int = time_stamp #TODO: choose normalization
        '''time_stamp represents the time step at which this node is reached and is thereby the cost'''
        self.control_input_used: ControlInput = control_input_used
        
    def get_parent(self) -> 'KinodynamicRRTNode':
        """Returns the parent node."""
        return self.parent
    
    def get_state(self) -> RobotState:
        """Returns the robot state at this node."""
        return self.robot_state
    
    #TODO: choose normalization
    def get_cost(self) -> int:
        """Returns the cost to reach this node."""
        return self.cost

    def get_control_input_used(self) -> ControlInput:
        """Returns the control input used to reach this node from its parent."""
        return self.control_input_used
    
    #TODO: choose normalization
    def set_cost(self) -> None:
        """Automatically sets the cost based on the parent's cost."""
        if self.parent is not None:
            self.cost = self.parent.get_cost() + 1
        else:
            self.cost = 0
            
    def set_cost(self, cost: int) -> None:
        """Sets the cost to reach this node."""
        self.cost = cost

    def set_parent(self, parent: 'KinodynamicRRTNode') -> None:
        """Sets the parent node."""
        self.parent = parent

    def set_control_input_used(self, control_input: ControlInput) -> None:
        """Sets the control input used to reach this node from its parent."""
        self.control_input_used = control_input

RRT_TIMESTEP: float = 0.2
'''Time step for each RRT expansion in seconds'''

def steer(initial_state: RobotState, u: ControlInput) -> Tuple[RobotState, ControlInput]:
    """Steers the robot from the initial state using control input u over a fixed time step.\n
    Returns the new robot state and the actual control inputs applied (which may be clamped to respect velocity/position limits)."""
    new_state: RobotState = RobotState(initial_state.angles_to_vector().copy(),
                                       initial_state.velocities_to_vector().copy())
    actual_controls: ControlInput = u.copy()
    for i in range(RobotState.DOF):
        # Update velocity with acceleration
        v0 = initial_state.velocities_to_vector()[i]
        s0 = initial_state.angles_to_vector()[i]
        new_velocity = v0 + u[i] * RRT_TIMESTEP

        # Check velocity limits, and clamp if necessary
        if new_velocity > RobotState.MAX_VELOCITIES[i] or new_velocity < -RobotState.MAX_VELOCITIES[i]:
            new_velocity = cp.sign(new_velocity) * RobotState.MAX_VELOCITIES[i]
            u[i] = (new_velocity - initial_state.velocities_to_vector()[i]) / RRT_TIMESTEP

        # Predict position using kinematics: s = s0 + v0*RRT_TIMESTEP + 0.5*a*RRT_TIMESTEP^2
        new_angle = s0 + v0 * RRT_TIMESTEP + 0.5 * u[i] * RRT_TIMESTEP * RRT_TIMESTEP
        if new_angle > RobotState.MAX_POSITIONS[i] or new_angle < -RobotState.MAX_POSITIONS[i]:
            # Choose the boundary in the direction of motion and compute the required velocity
            bound = cp.sign(new_velocity) * RobotState.MAX_POSITIONS[i]
            new_velocity = (2.0 * (bound - s0) / RRT_TIMESTEP) - v0
            u[i] = (new_velocity - v0) / RRT_TIMESTEP
            new_angle = s0 + v0 * RRT_TIMESTEP + 0.5 * u[i] * RRT_TIMESTEP * RRT_TIMESTEP
        
        # Update the state
        new_state.angles_to_vector()[i] = new_angle
        new_state.velocities_to_vector()[i] = new_velocity

    return new_state, actual_controls

# TODO: Consider only using the angles and not the velocities
class KRRT_KDTree:
    """KD-Tree implementation for Kinodynamic RRT nodes."""
    def __init__(self, nodes: List[KinodynamicRRTNode]) -> None:
        self.nodes = nodes
        states: List[RobotKinodynamicState] = cp.array([node.to_vector() for node in nodes])
        self._kd_tree = KDTree(states)

    # TODO: consider using custom weights for distance metric
    def query(self, sample_state: KinodynamicRRTNode, k: int = 1) -> List[Tuple[KinodynamicRRTNode, float]]:
        """Finds the nearest node in the RRT to the given sample state based on Euclidean distance in joint space."""
        sample_vector = sample_state.robot_state.to_vector().reshape(1, -1) # Shape (1, 12) for the kd-tree query
        dist, indexes = self._kd_tree.query(sample_vector, k=k)
        # Normalize outputs to 1D to support both k==1 and k>1
        if k == 1:
            return [(self.nodes[int(indexes[0])], float(dist[0]))]
        else:
            return [(self.nodes[i], float(dist[0][j])) for j, i in enumerate(indexes[0])]

    # TODO: consider using custom weights for distance metric
    def query_ball_point(self, sample_state: KinodynamicRRTNode, r: float) -> List[KinodynamicRRTNode]:
        """Finds all nodes within radius r of the given sample state."""
        sample_vector = sample_state.robot_state.to_vector().reshape(1, -1) # Shape (1, 12) for the kd-tree query_ball_point
        indices = self._kd_tree.query_ball_point(sample_vector, r)
        return [self.nodes[i] for i in indices[0]]
    
    def add_node(self, node: KinodynamicRRTNode) -> None:
        """Adds a new node to the KD-Tree."""
        self.nodes.append(node)
        states: List[RobotKinodynamicState] = cp.array([n.to_vector() for n in self.nodes])
        self._kd_tree = KDTree(states)

# TODO: consider adding a bias search
class KRRT_Star_Calculator:
    """Kinodynamic RRT* calculations implementation."""
    def __init__(self, initial_angles: RobotAnglesVector, goal_angles: RobotAnglesVector,
                 initial_velocities: RobotAnglesVector = cp.zeros(6), goal_velocities: RobotAnglesVector = cp.zeros(6)) -> None:
        self.start_node: KinodynamicRRTNode = KinodynamicRRTNode(RobotState(initial_angles, initial_velocities), 0)
        self.goal_node: KinodynamicRRTNode = KinodynamicRRTNode(RobotState(goal_angles, goal_velocities), 0)
        '''The goal node of the RRT.'''
        self.nodes: KRRT_KDTree = KRRT_KDTree([self.start_node])
        '''The KD-Tree of nodes in the RRT.'''
        self.N = 1
        '''The number of nodes in the RRT.'''
        
    @staticmethod
    def from_states(initial_state: RobotState, goal_state: RobotState) -> "KRRT_Star_Calculator":
        return KRRT_Star_Calculator(initial_state.angles_to_vector(), goal_state.angles_to_vector(), initial_state.velocities_to_vector(), goal_state.velocities_to_vector())

    # ------------ Nearest Neighbor Methods -------------
    def get_nearest_node(self, sample_state: RobotState) -> KinodynamicRRTNode:
        """Finds the nearest node in the RRT to the given sample state based on Euclidean distance in joint space."""
        return self.nodes.query(KinodynamicRRTNode(sample_state, 0), k=1)[0][0]

    def get_connection_radius(self) -> float:
        """Computes the connection radius for RRT* based on the number of nodes."""
        # Using a common formula for RRT* connection radius #TODO: change this scaling factor if needed
        gamma_rrt_star: float = 2.0 * (1 + 1.0 / (2 * RobotState.DOF)) ** (1.0 / (2 * RobotState.DOF))
        '''Scaling factor for the connection radius.'''
        '''The number of nodes in the RRT.'''
        return float(gamma_rrt_star * ((cp.log(self.N) / self.N) ** (1.0 / (2 * RobotState.DOF))))

    # ------------- Sampling Methods -------------
    def get_biased_control(self, from_state: RobotState) -> ControlInput:
        """Generates a control input that biases the robot towards the goal state."""
        direction: RobotAnglesVector = self.goal_node.robot_state.angles_to_vector() - from_state.angles_to_vector()
        norm: float = cp.linalg.norm(direction)
        if norm == 0:
            return cp.zeros(RobotState.DOF)
        unit_direction: RobotAnglesVector = direction / norm
        # Scale by max accelerations to get a strong bias towards the goal
        biased_control: ControlInput = unit_direction * RobotState.MAX_ACCELERATIONS
        return biased_control

    @staticmethod
    def sample_control() -> ControlInput:
        """Samples a random control input within the robot's acceleration limits."""
        # Return shape (6,) with per-joint limits: [-MAX_ACCEL[i], +MAX_ACCEL[i]]
        return RobotState.MAX_ACCELERATIONS * cp.random.uniform(-1.0, 1.0, size=(RobotState.DOF,), dtype=cp.float64)

    # ------------- Validity Checking Methods -------------
    @NotImplementedError
    # TODO: Implement collision checking transitioning between two robot states
    def check_collision(self, state1: RobotState, state2: RobotState) -> bool:
        """Checks for collisions between two robot states. Returns True if a collision is detected."""
        raise NotImplementedError("Collision checking not implemented yet.")

    def check_feasibility(self, from_state: RobotState, to_state: RobotState) -> bool:
        """Checks if the transition from from_state to to_state is feasible."""
        states_diff: RobotAnglesVector = to_state.angles_to_vector() - from_state.angles_to_vector()
        velocities_diff: RobotVelocitiesVector = to_state.velocities_to_vector() - from_state.velocities_to_vector()
        # Check if the required acceleration exceeds max limits
        for i in range(RobotState.DOF):
            required_acceleration: float = (velocities_diff[i]) / RRT_TIMESTEP
            if abs(required_acceleration) > RobotState.MAX_ACCELERATIONS[i]:
                return False

            # Using kinematic equation: Δs = v0*t + 0.5*a*t^2
            # Here states_diff[i] = to_state.angle[i] - from_state.angle[i] = Δs
            if not states_diff[i] == from_state.velocities_to_vector()[i] * RRT_TIMESTEP + 0.5 * required_acceleration * (RRT_TIMESTEP ** 2):
                return False

        return True
   
    def check_connection_validity(self, from_state: RobotState, to_state: RobotState) -> bool:
        """Checks if the connection between two states is valid (collision-free and feasible)."""
        if not self.check_feasibility(from_state, to_state):
            return False
        if self.check_collision(from_state, to_state):
            return False
        return True


    def check_rewiring_feasibility(self, from_node: KinodynamicRRTNode, to_node: KinodynamicRRTNode, final_node: bool = False) -> Tuple[bool, ControlInput]:  
        """Checks if rewiring from from_node to to_node is feasible.\n
            Returns a tuple (feasible: bool, desired_accelerations: ControlInput)."""
        # States
        start_state: RobotState = from_node.get_state()
        end_state: RobotState = to_node.get_state()
        end_parent_state: RobotState = to_node.get_parent().get_state()
        # Costs
        start_state_cost: float = from_node.get_cost()
        end_state_cost: float = to_node.get_cost()
        start_parent_cost: float = from_node.get_parent().get_cost()
        end_parent_cost: float = to_node.get_parent().get_cost()
            
        state_diffs: RobotAnglesVector = end_state.angles_to_vector() - start_state.angles_to_vector()
        start_vel: RobotVelocitiesVector = start_state.velocities_to_vector()
        max_acc: RobotAnglesVector = RobotState.MAX_ACCELERATIONS
        max_vel: RobotVelocitiesVector = RobotState.MAX_VELOCITIES
        tol_pos: float = 1e-6
        feasible: bool = True
        desired_accelerations: ControlInput = cp.zeros(RobotState.DOF)
        
        for i in range(RobotState.DOF):
            delta_s: float = state_diffs[i]
            v0: float = start_vel[i]

            # Acceleration required to achieve the position delta exactly
            a_req: float = 2.0 * (delta_s - v0 * RRT_TIMESTEP) / (RRT_TIMESTEP * RRT_TIMESTEP)
            v1_req: float = v0 + a_req * RRT_TIMESTEP

            acc_ok: bool = abs(a_req) <= max_acc[i]
            vel_ok: bool = abs(v1_req) <= max_vel[i]

            # Position consistency check
            pos_ok: bool = abs(delta_s - (v0 * RRT_TIMESTEP + 0.5 * a_req * RRT_TIMESTEP * RRT_TIMESTEP)) <= tol_pos
            if not acc_ok or not vel_ok or not pos_ok:
                return False, None

            if final_node:
                # Require acceleration that ends at zero velocity
                a_req: float = -v0 / RRT_TIMESTEP
                acc_ok: bool = abs(a_req) <= max_acc[i]
                if not acc_ok:
                    return False, None
                # Position consistency check for stopping at zero velocity
                if not abs(delta_s - (v0 * RRT_TIMESTEP + 0.5 * a_req * RRT_TIMESTEP * RRT_TIMESTEP)) <= tol_pos:
                    return False, None

            desired_accelerations[i] = a_req

        return feasible, desired_accelerations      
    
    # ------------ Node Management -------------

    def add_node(self, node: KinodynamicRRTNode) -> None:
        """Adds a new node to the RRT.\n
        Assumes the parent has already been set."""
        self.nodes.add_node(node)  # Automatically set cost based on parent
        self.rewire(node)
        self.N += 1 ## Increment the node count

    def rewire(self, new_node: KinodynamicRRTNode) -> bool:
        """Rewires the RRT to optimize paths through the new node. Returns True if any rewiring occurred."""
        neighboring_nodes: List[KinodynamicRRTNode] = self.nodes.query_ball_point(new_node, self.get_connection_radius())
        current_node_cost: float = new_node.get_cost()
        rewired: bool = False
        '''Indicates if any rewiring occurred'''
        for neighbor in neighboring_nodes:
            potential_cost = new_node.get_cost() + 1.  # Assuming unit cost for each step #TODO: make sure this works with the timestamps
            #Check rewiring possibility from new_node to neighbor and what controls are needed for that
            connection_valid, needed_controls = self.check_rewiring_feasibility(new_node, neighbor)
            if connection_valid and not self.check_collision(new_node.get_state(), neighbor.get_state()) and potential_cost < neighbor.get_cost():
                neighbor.set_parent(new_node)
                neighbor.set_control_input_used(needed_controls)
                neighbor.set_cost(potential_cost)
                rewired = True

            #Check rewiring possibility from neighbor to new_node and what controls are needed for that
            potential_cost = neighbor.get_cost() + 1.
            connection_valid, needed_controls = self.check_rewiring_feasibility(neighbor, new_node)
            if connection_valid and not self.check_collision(neighbor.get_state(), new_node.get_state()) and potential_cost < current_node_cost:
                new_node.set_parent(neighbor)
                new_node.set_cost(potential_cost)
                new_node.set_control_input_used(needed_controls)
                current_node_cost = potential_cost
                rewired = True
        return rewired

#--------------------------------------------------------- Global vars
destination: RobotAnglesVector = None
'''Current robot destination as read from /joint_destination'''
destination_outdated: bool = False
'''Flag indicating if the destination has been updated'''
pose_seq: HumanPoseSequence = None
'''Current human pose sequence as read from /joint_array'''
body_links: List[Link] = []
'''List of human body links extracted from the current human pose'''
robot_joint_angles: RobotAnglesVector = cp.zeros(6)
'''Robot joint positions as in the angle it is currently in as a cupy array of shape (6,)'''

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

######### ROS2 Nodes #########

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
        pose_seq = cp.array(msg.data).reshape((1, 15, 3))
        body_links = extract_links_gpu(pose_seq[0])

class APFVisualizationPublisher(Node):
    """Publishes arrow markers showing APF forces from robot links to human body parts."""
    
    def __init__(self) -> None:
        super().__init__('apf_visualization_publisher')
        self.publisher_ = self.create_publisher(MarkerArray, '/apf_forces', 10)
        self.timer = self.create_timer(0.05, self.publish_apf_markers)  # 20Hz update rate
        
    def publish_apf_markers(self) -> None:
        """Create and publish arrow markers representing APF forces."""
        global robot_joint_angles, body_links
        
        if len(body_links) == 0:
            self.get_logger().debug("No body links available")
            return
            
        marker_array = MarkerArray()
        marker_id = 0
        
        # Get robot joint positions (centers of links, in mm)
        robot_joint_positions = dh_transform_batch(robot_joint_angles)  # (6, 3) in mm
        
        # Create robot link centers (midpoints between consecutive joints)
        robot_link_centers = []
        for i in range(len(robot_joint_positions) - 1):
            link_center = (robot_joint_positions[i] + robot_joint_positions[i+1]) / 2.0
            robot_link_centers.append(link_center / 1000.0)  # Convert to meters
        
        # For each human body link
        for link_start, link_end, radius in body_links:
            # Find closest robot link center to this human link
            min_dist = float('inf')
            closest_robot_link_center = None
            closest_human_pt = None
            
            for robot_link_center in robot_link_centers:
                # Distance from robot link center to human link capsule
                link_vec = link_end - link_start
                link_len = float(cp.linalg.norm(link_vec))
                
                if link_len < 1e-6:
                    projection = link_start
                    dist = float(cp.linalg.norm(robot_link_center - link_start)) - float(radius)
                else:
                    # Project robot point onto human link
                    v = robot_link_center - link_start
                    t = float(cp.dot(v, link_vec)) / link_len
                    t = max(0.0, min(t, link_len))
                    projection = link_start + (t / link_len) * link_vec
                    dist = float(cp.linalg.norm(robot_link_center - projection)) - float(radius)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_robot_link_center = robot_link_center
                    closest_human_pt = projection
            
            # Calculate APF magnitude for this link
            if min_dist <= 0:
                apf_magnitude = 1000.0  # Collision
            elif min_dist > 2.0:  # Beyond threshold (increased from 1.0 to 2.0m)
                continue  # Skip this link
            else:
                # Exponential repulsion
                safe_d = max(min_dist, 0.01)
                apf_magnitude = 80.0 * float(cp.exp(-safe_d / 0.1))
            
            # Create arrow marker from robot point to human surface
            marker = Marker()
            marker.header.frame_id = "base"  # Use same frame as human markers
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "apf_forces"
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Start point (robot link center) - already in meters
            start_pt = Point()
            start_pt.x = float(closest_robot_link_center[0])
            start_pt.y = float(closest_robot_link_center[1])
            start_pt.z = float(closest_robot_link_center[2])
            
            # End point (closest point on human capsule surface)
            end_pt = Point()
            end_pt.x = float(closest_human_pt[0])
            end_pt.y = float(closest_human_pt[1])
            end_pt.z = float(closest_human_pt[2])
            
            marker.points = [start_pt, end_pt]
            
            # Arrow size
            marker.scale.x = 0.01  # Shaft diameter
            marker.scale.y = 0.02  # Head diameter
            marker.scale.z = 0.03  # Head length
            
            # Color based on APF magnitude relative to threshold
            color = ColorRGBA()
            if apf_magnitude > apf_th:
                color.r, color.g, color.b = 1.0, 0.0, 0.0  # Red - exceeds threshold
            elif apf_magnitude > apf_th * 0.5:
                color.r, color.g, color.b = 1.0, 0.5, 0.0  # Orange - high (50-100% of threshold)
            elif apf_magnitude > apf_th * 0.25:
                color.r, color.g, color.b = 1.0, 1.0, 0.0  # Yellow - medium (25-50% of threshold)
            else:
                color.r, color.g, color.b = 0.0, 1.0, 0.0  # Green - low (< 25% of threshold)
            color.a = 0.8
            marker.color = color
            
            # Set lifetime to 0 for persistent markers (they'll be updated by the timer)
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 0
            
            marker_array.markers.append(marker)
            marker_id += 1
        
        # Delete old markers if we have fewer now
        if marker_id < 20:  # Assuming max 20 markers
            for i in range(marker_id, 20):
                delete_marker = Marker()
                delete_marker.header.frame_id = "base"
                delete_marker.ns = "apf_forces"
                delete_marker.id = i
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)
        
        self.publisher_.publish(marker_array)

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
        self.get_logger().info(f"Published trajectory to: {point.positions}")

    def main_loop(self) -> None:
        global robot_joint_angles, body_links, published, destination, destination_outdated
        time.sleep(0.5)  # Wait for other nodes to initialize
        #Wait until there is a destinaition to go to
        while destination is None:
            self.get_logger().warning(str(destination))
            time.sleep(0.1)

        current: RobotAnglesVector = robot_joint_angles.copy()
        self.send_trajectory(current)
        path: List[RobotAnglesVector] = arrt(current, destination, 200)
        self.get_logger().info(f"After initial planning, path length: {len(path)}")
        step: int = 1
        look_ahead_steps: int = 3 # How many steps ahead to check for APF threshold exceedance
        while rclpy.ok():
            apf: float = APF_gpu(robot_joint_angles, body_links)
            temp: int = step

            # Check whether destination has been updated
            if destination_outdated:
                self.get_logger().info("Destination updated, replanning path.")
                path = arrt(robot_joint_angles, destination, 200)
                self.get_logger().info(f"Replanned path length: {len(path)}")
                self.send_trajectory(path[1])
                step = 2
                destination_outdated = False
                continue

            # Look ahead along the path to see if APF exceeds threshold
            while apf < apf_th and (temp < len(path) and temp - step < look_ahead_steps):
                apf = max(apf, APF_gpu(path[temp], body_links))
                temp += 1

            # Replan if APF threshold exceeded
            if apf > apf_th:
                self.send_trajectory(robot_joint_angles) # Stop movement
                self.get_logger().warning(f"Replanning path due to apf threshold length: {len(path)}")
                path = arrt(robot_joint_angles, destination, 200)
                self.get_logger().info(f"Replanned path due to apf threshold length: {len(path)}")
                self.send_trajectory(path[1])
                step = 2
                continue

            dist_from_published = cp.linalg.norm(((robot_joint_angles - published + cp.pi) % (2 * cp.pi)) - cp.pi)
            # Move to next point if close enough to last published point
            if dist_from_published < 0.1 and step < len(path):
                next_pos = path[step]
                self.send_trajectory(next_pos)
                step += 1
            
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

    return cp.stack(positions) * 1000  # shape: (6, 3) in mm


# TODO: understand what this returns, also because this reaches capsule_contrib_batch which has unclarity in types
def get_full_link_points_gpu(joints: RobotJointPositions, n: int = 5):
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
        if d_norm < 1e-6:
            continue
        v = points - p1  # TODO: the types mismatch. points is 6x3, but p1 is 3x1
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
    # self.get_logger().info(f"APF computation time: {time.time()-t1:.6f} seconds")
    return ccb


# ----------------- A-RRT* Planning Function -----------------
def arrt(q_start: RobotState, q_goal: RobotState, n_nodes: int = 100):
    start_t = time.time()
    n_explored: int = 0
    n_used: int = 0

    start_tree: List[RRTNode] = [RRTNode(q_start, 0)]
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
            # self.get_logger().info("adding nodes to goal tree")
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
    apf_viz_publisher = APFVisualizationPublisher()

    executor = MultiThreadedExecutor()
    while not pose_listener.ready and rclpy.ok():
        rclpy.spin_once(pose_listener, timeout_sec=0.1)

    executor.add_node(pose_listener)
    executor.add_node(joint_reader)
    executor.add_node(destination_reader)
    executor.add_node(trajectory_publisher)
    executor.add_node(apf_viz_publisher)

    try:
        executor.spin()
    finally:
        joint_reader.destroy_node()
        destination_reader.destroy_node()
        pose_listener.destroy_node()
        trajectory_publisher.destroy_node()
        apf_viz_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

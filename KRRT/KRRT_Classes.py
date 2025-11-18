import cupy as cp
from cupyx.scipy.spatial import KDTree
from typing import List, Tuple
from classes_and_types import *

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

import cupy as cp
from cupyx.scipy.spatial import KDTree
from typing import List, Tuple, Annotated, TypeAlias
from cupy.typing import NDArray

#---------------------------------------------------------- Type Aliasing for 2DoF
PositionVector2D: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (2,) representing the x,y position of the robot'''
VelocityVector2D: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (2,) representing the x,y velocities of the robot'''
KinodynamicState2D: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (4,) representing the x,y positions and x,y velocities of the robot'''
ControlInput2D: TypeAlias = Annotated[NDArray[cp.float64], cp.ndarray]
'''Shape : (2,) A control input is represented as a vector of accelerations in x,y directions'''

#---------------------------------------------------------- Robot State Class
class RobotState2D:
    """Represents the kinodynamic state of a 2DoF robot: x,y positions and x,y velocities."""
    #Static fields (class variables)
    DOF: int = 2
    '''Degrees of freedom'''
    MAX_POSITIONS: PositionVector2D = cp.array([10.0, 10.0])
    '''Max positions in x,y directions (meters)'''
    MAX_VELOCITIES: VelocityVector2D = cp.array([2.0, 2.0])
    '''Max velocities in x,y directions (m/s)'''
    MAX_ACCELERATIONS: PositionVector2D = cp.array([3.0, 3.0])
    '''Max accelerations in x,y directions (m/s^2)'''
    MIN_POSITIONS: PositionVector2D = cp.array([-10.0, -10.0])
    '''Min positions in x,y directions (meters)'''

    def __init__(self, positions: PositionVector2D, velocities: VelocityVector2D) -> None:
        self.positions: PositionVector2D = positions.copy()
        self.velocities: VelocityVector2D = velocities.copy()

    ############Methods to convert to vectors
    def positions_to_vector(self) -> PositionVector2D:
        """returns the x,y positions as a cupy array of shape (2,)"""
        return self.positions
    
    def velocities_to_vector(self) -> VelocityVector2D:
        """returns the x,y velocities as a cupy array of shape (2,)"""
        return self.velocities

    def to_vector(self) -> KinodynamicState2D:
        """returns the positions and velocities concatenated as a single cupy array of shape (4,)"""
        return cp.concatenate((self.positions_to_vector(), self.velocities_to_vector()))
    
    def copy(self) -> 'RobotState2D':
        """Returns a copy of the robot state."""
        return RobotState2D(self.positions.copy(), self.velocities.copy())

#------------------------------------------------------- RRT Functions
class KinodynamicRRTNode2D:
    """A node in the kinodynamic RRT for 2DoF robot. Holds the robot state, parent node, and cost to reach this node."""
    def __init__(self, state: RobotState2D, time_stamp: int, control_input_used: ControlInput2D = None) -> None:
        self.robot_state: RobotState2D = state
        '''The robot state at this node'''
        self.parent: 'KinodynamicRRTNode2D' = None
        '''Parent node in the RRT'''
        self.cost: int = time_stamp #TODO: choose normalization
        '''time_stamp represents the time step at which this node is reached and is thereby the cost'''
        self.control_input_used: ControlInput2D = control_input_used
        
    def get_parent(self) -> 'KinodynamicRRTNode2D':
        """Returns the parent node."""
        return self.parent
    
    def get_state(self) -> RobotState2D:
        """Returns the robot state at this node."""
        return self.robot_state
    
    #TODO: choose normalization
    def get_cost(self) -> int:
        """Returns the cost to reach this node."""
        return self.cost

    def get_control_input_used(self) -> ControlInput2D:
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

    def set_parent(self, parent: 'KinodynamicRRTNode2D') -> None:
        """Sets the parent node."""
        self.parent = parent

    def set_control_input_used(self, control_input: ControlInput2D) -> None:
        """Sets the control input used to reach this node from its parent."""
        self.control_input_used = control_input
    
    def to_vector(self) -> KinodynamicState2D:
        """Returns the kinodynamic state as a vector."""
        return self.robot_state.to_vector()

RRT_TIMESTEP: float = 0.2
'''Time step for each RRT expansion in seconds'''

def steer_2d(initial_state: RobotState2D, u: ControlInput2D) -> Tuple[RobotState2D, ControlInput2D]:
    """Steers the 2DoF robot from the initial state using control input u over a fixed time step.\n
    Returns the new robot state and the actual control inputs applied (which may be clamped to respect velocity/position limits)."""
    new_state: RobotState2D = RobotState2D(initial_state.positions_to_vector().copy(),
                                           initial_state.velocities_to_vector().copy())
    actual_controls: ControlInput2D = u.copy()
    
    for i in range(RobotState2D.DOF):
        # Update velocity with acceleration
        v0 = initial_state.velocities_to_vector()[i]
        s0 = initial_state.positions_to_vector()[i]
        new_velocity = v0 + u[i] * RRT_TIMESTEP

        # Check velocity limits, and clamp if necessary
        if new_velocity > RobotState2D.MAX_VELOCITIES[i] or new_velocity < -RobotState2D.MAX_VELOCITIES[i]:
            new_velocity = cp.sign(new_velocity) * RobotState2D.MAX_VELOCITIES[i]
            actual_controls[i] = (new_velocity - initial_state.velocities_to_vector()[i]) / RRT_TIMESTEP

        # Predict position using kinematics: s = s0 + v0*RRT_TIMESTEP + 0.5*a*RRT_TIMESTEP^2
        new_position = s0 + v0 * RRT_TIMESTEP + 0.5 * actual_controls[i] * RRT_TIMESTEP * RRT_TIMESTEP
        
        # Check position limits
        if new_position > RobotState2D.MAX_POSITIONS[i]:
            # Hit upper boundary - compute required velocity to reach boundary
            bound = RobotState2D.MAX_POSITIONS[i]
            new_velocity = (2.0 * (bound - s0) / RRT_TIMESTEP) - v0
            actual_controls[i] = (new_velocity - v0) / RRT_TIMESTEP
            new_position = s0 + v0 * RRT_TIMESTEP + 0.5 * actual_controls[i] * RRT_TIMESTEP * RRT_TIMESTEP
        elif new_position < RobotState2D.MIN_POSITIONS[i]:
            # Hit lower boundary - compute required velocity to reach boundary
            bound = RobotState2D.MIN_POSITIONS[i]
            new_velocity = (2.0 * (bound - s0) / RRT_TIMESTEP) - v0
            actual_controls[i] = (new_velocity - v0) / RRT_TIMESTEP
            new_position = s0 + v0 * RRT_TIMESTEP + 0.5 * actual_controls[i] * RRT_TIMESTEP * RRT_TIMESTEP
        
        # Update the state
        new_state.positions_to_vector()[i] = new_position
        new_state.velocities_to_vector()[i] = new_velocity

    return new_state, actual_controls

# TODO: Consider only using the positions and not the velocities
class KRRT_KDTree2D:
    """KD-Tree implementation for Kinodynamic RRT nodes in 2D."""
    def __init__(self, nodes: List[KinodynamicRRTNode2D]) -> None:
        self.nodes = nodes
        states: List[KinodynamicState2D] = cp.array([node.to_vector() for node in nodes])
        self._kd_tree = KDTree(states)

    # TODO: consider using custom weights for distance metric
    def query(self, sample_state: KinodynamicRRTNode2D, k: int = 1) -> List[Tuple[KinodynamicRRTNode2D, float]]:
        """Finds the nearest node in the RRT to the given sample state based on Euclidean distance."""
        sample_vector = sample_state.robot_state.to_vector().reshape(1, -1) # Shape (1, 4) for the kd-tree query
        dist, indexes = self._kd_tree.query(sample_vector, k=k)
        # Normalize outputs to 1D to support both k==1 and k>1
        if k == 1:
            return [(self.nodes[int(indexes[0])], float(dist[0]))]
        else:
            return [(self.nodes[i], float(dist[0][j])) for j, i in enumerate(indexes[0])]

    # TODO: consider using custom weights for distance metric
    def query_ball_point(self, sample_state: KinodynamicRRTNode2D, r: float) -> List[KinodynamicRRTNode2D]:
        """Finds all nodes within radius r of the given sample state."""
        sample_vector = sample_state.robot_state.to_vector().reshape(1, -1) # Shape (1, 4) for the kd-tree query_ball_point
        indices = self._kd_tree.query_ball_point(sample_vector, r)
        return [self.nodes[i] for i in indices[0]]
    
    def add_node(self, node: KinodynamicRRTNode2D) -> None:
        """Adds a new node to the KD-Tree."""
        self.nodes.append(node)
        states: List[KinodynamicState2D] = cp.array([n.to_vector() for n in self.nodes])
        self._kd_tree = KDTree(states)

# TODO: consider adding a bias search
class KRRT_Star_Calculator2D:
    """Kinodynamic RRT* calculations implementation for 2DoF robot."""
    def __init__(self, initial_positions: PositionVector2D, goal_positions: PositionVector2D,
                 initial_velocities: VelocityVector2D = cp.zeros(2), goal_velocities: VelocityVector2D = cp.zeros(2)) -> None:
        self.start_node: KinodynamicRRTNode2D = KinodynamicRRTNode2D(RobotState2D(initial_positions, initial_velocities), 0)
        self.goal_node: KinodynamicRRTNode2D = KinodynamicRRTNode2D(RobotState2D(goal_positions, goal_velocities), 0)
        '''The goal node of the RRT.'''
        self.nodes: KRRT_KDTree2D = KRRT_KDTree2D([self.start_node])
        '''The KD-Tree of nodes in the RRT.'''
        self.N = 1
        '''The number of nodes in the RRT.'''
        
    @staticmethod
    def from_states(initial_state: RobotState2D, goal_state: RobotState2D) -> "KRRT_Star_Calculator2D":
        return KRRT_Star_Calculator2D(initial_state.positions_to_vector(), goal_state.positions_to_vector(), 
                                       initial_state.velocities_to_vector(), goal_state.velocities_to_vector())

    # ------------ Nearest Neighbor Methods -------------
    def get_nearest_node(self, sample_state: RobotState2D) -> KinodynamicRRTNode2D:
        """Finds the nearest node in the RRT to the given sample state based on Euclidean distance."""
        return self.nodes.query(KinodynamicRRTNode2D(sample_state, 0), k=1)[0][0]

    def get_connection_radius(self) -> float:
        """Computes the connection radius for RRT* based on the number of nodes."""
        # Using a common formula for RRT* connection radius #TODO: change this scaling factor if needed
        gamma_rrt_star: float = 2.0 * (1 + 1.0 / (2 * RobotState2D.DOF)) ** (1.0 / (2 * RobotState2D.DOF))
        '''Scaling factor for the connection radius.'''
        '''The number of nodes in the RRT.'''
        return float(gamma_rrt_star * ((cp.log(self.N) / self.N) ** (1.0 / (2 * RobotState2D.DOF))))

    # ------------- Sampling Methods -------------
    def get_biased_control(self, from_state: RobotState2D) -> ControlInput2D:
        """Generates a control input that biases the robot towards the goal state."""
        direction: PositionVector2D = self.goal_node.robot_state.positions_to_vector() - from_state.positions_to_vector()
        norm: float = cp.linalg.norm(direction)
        if norm == 0:
            return cp.zeros(RobotState2D.DOF)
        unit_direction: PositionVector2D = direction / norm
        # Scale by max accelerations to get a strong bias towards the goal
        biased_control: ControlInput2D = unit_direction * RobotState2D.MAX_ACCELERATIONS
        return biased_control

    @staticmethod
    def sample_control() -> ControlInput2D:
        """Samples a random control input within the robot's acceleration limits."""
        # Return shape (2,) with per-direction limits: [-MAX_ACCEL[i], +MAX_ACCEL[i]]
        return RobotState2D.MAX_ACCELERATIONS * cp.random.uniform(-1.0, 1.0, size=(RobotState2D.DOF,), dtype=cp.float64)

    # ------------- Validity Checking Methods -------------
    @staticmethod
    def check_collision(state1: RobotState2D, state2: RobotState2D) -> bool:
        """Checks for collisions between two robot states. Returns True if a collision is detected.
        For 2D point robot, this is a placeholder - implement with actual obstacle checking."""
        # TODO: Implement collision checking with obstacles
        return False

    def check_feasibility(self, from_state: RobotState2D, to_state: RobotState2D) -> bool:
        """Checks if the transition from from_state to to_state is feasible."""
        states_diff: PositionVector2D = to_state.positions_to_vector() - from_state.positions_to_vector()
        velocities_diff: VelocityVector2D = to_state.velocities_to_vector() - from_state.velocities_to_vector()
        # Check if the required acceleration exceeds max limits
        for i in range(RobotState2D.DOF):
            required_acceleration: float = (velocities_diff[i]) / RRT_TIMESTEP
            if abs(required_acceleration) > RobotState2D.MAX_ACCELERATIONS[i]:
                return False

            # Using kinematic equation: Δs = v0*t + 0.5*a*t^2
            # Here states_diff[i] = to_state.position[i] - from_state.position[i] = Δs
            expected_displacement = from_state.velocities_to_vector()[i] * RRT_TIMESTEP + 0.5 * required_acceleration * (RRT_TIMESTEP ** 2)
            if not abs(states_diff[i] - expected_displacement) < 1e-6:
                return False

        return True
   
    def check_connection_validity(self, from_state: RobotState2D, to_state: RobotState2D) -> bool:
        """Checks if the connection between two states is valid (collision-free and feasible)."""
        if not self.check_feasibility(from_state, to_state):
            return False
        if self.check_collision(from_state, to_state):
            return False
        return True


    def check_rewiring_feasibility(self, from_node: KinodynamicRRTNode2D, to_node: KinodynamicRRTNode2D, final_node: bool = False) -> Tuple[bool, ControlInput2D]:  
        """Checks if rewiring from from_node to to_node is feasible.\n
            Returns a tuple (feasible: bool, desired_accelerations: ControlInput2D)."""
        # States
        start_state: RobotState2D = from_node.get_state()
        end_state: RobotState2D = to_node.get_state()
            
        state_diffs: PositionVector2D = end_state.positions_to_vector() - start_state.positions_to_vector()
        start_vel: VelocityVector2D = start_state.velocities_to_vector()
        max_acc: PositionVector2D = RobotState2D.MAX_ACCELERATIONS
        max_vel: VelocityVector2D = RobotState2D.MAX_VELOCITIES
        tol_pos: float = 1e-6
        feasible: bool = True
        desired_accelerations: ControlInput2D = cp.zeros(RobotState2D.DOF)
        
        for i in range(RobotState2D.DOF):
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

    def add_node(self, node: KinodynamicRRTNode2D) -> None:
        """Adds a new node to the RRT.\n
        Assumes the parent has already been set."""
        self.nodes.add_node(node)  # Automatically set cost based on parent
        self.rewire(node)
        self.N += 1 ## Increment the node count

    def rewire(self, new_node: KinodynamicRRTNode2D) -> bool:
        """Rewires the RRT to optimize paths through the new node. Returns True if any rewiring occurred."""
        neighboring_nodes: List[KinodynamicRRTNode2D] = self.nodes.query_ball_point(new_node, self.get_connection_radius())
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

    # ------------ Path Retrieval -------------
   
    def get_path_and_controls(self, node: KinodynamicRRTNode2D) -> Tuple[List[KinodynamicRRTNode2D], List[ControlInput2D]]:
        """Returns the path and control sequence from the start node to the given node."""
        path: List[KinodynamicRRTNode2D] = []
        controls: List[ControlInput2D] = []
        current_node = node
        while current_node is not None:
            path.append(current_node)
            if current_node.get_control_input_used() is not None:
                controls.append(current_node.get_control_input_used())
            current_node = current_node.get_parent()
        return path[::-1], controls[::-1]
#!/usr/bin/env python3
"""
Standalone OMPL Kinodynamic RRT* implementation for 2D point robot.
Run with: conda activate master_project && python ompl_kinodynamic_rrt_star_2d.py

OMPL Overview:
--------------
OMPL (Open Motion Planning Library) is a library for sampling-based motion planning.
It provides tools for planning in both geometric spaces (just positions) and 
kinodynamic spaces (positions + velocities, with dynamics constraints).

Key OMPL Concepts Used Here:
- StateSpace: Defines what a "state" is (e.g., x, y, vx, vy)
- ControlSpace: Defines what control inputs are available (e.g., ax, ay)
- SpaceInformation: Combines state/control spaces with validity checking
- StatePropagator: Function that simulates how controls affect state over time
- Planner (RRT): Algorithm that builds a tree of feasible trajectories

Features of This Implementation:
- 2D state space: (x, y, vx, vy) - position and velocity in 2D
- Control space: (ax, ay) - accelerations in x and y directions
- Velocity and acceleration limits enforced during propagation
- RRT algorithm for kinodynamic planning (can easily switch to RRT*, SST, etc.)
- Obstacle avoidance with circular obstacles
"""

from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from ompl import base as ob      # OMPL base module: state spaces, problem definitions
from ompl import control as oc   # OMPL control module: control spaces, kinodynamic planning
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

class RobotConfig:
    """Configuration parameters for 2D kinodynamic robot.
    
    These parameters define the workspace bounds and physical limits of the robot.
    Adjust these to match your specific robot or simulation requirements.
    """
    
    # -------- Workspace Bounds --------
    # Position limits define the rectangular workspace (meters)
    MIN_X = -10.0  # Minimum x position (left boundary)
    MAX_X = 10.0   # Maximum x position (right boundary)
    MIN_Y = -10.0  # Minimum y position (bottom boundary)
    MAX_Y = 10.0   # Maximum y position (top boundary)
    
    # -------- Velocity Limits --------
    # Maximum velocities in each direction (m/s)
    # These limits are enforced during state propagation
    MAX_VX = 2.0  # Maximum velocity in x direction
    MAX_VY = 2.0  # Maximum velocity in y direction
    
    # -------- Acceleration Limits --------
    # Maximum accelerations (control inputs) in each direction (m/s^2)
    # These represent the physical limits of what the robot can do
    MAX_AX = 3.0  # Maximum acceleration in x direction
    MAX_AY = 3.0  # Maximum acceleration in y direction
    
    # -------- Integration Parameters --------
    # PROPAGATION_STEP_SIZE: Time step for numerical integration (seconds)
    # Smaller values = more accurate but slower planning
    # Larger values = faster but less accurate
    PROPAGATION_STEP_SIZE = 0.05  # 50ms per integration step
    
    # -------- Control Duration Range --------
    # These parameters determine how long each sampled control is applied
    # Measured in number of propagation steps
    # MIN_CONTROL_DURATION * PROPAGATION_STEP_SIZE = minimum control time
    # MAX_CONTROL_DURATION * PROPAGATION_STEP_SIZE = maximum control time
    MIN_CONTROL_DURATION = 5   # Minimum: 5 steps = 0.25 seconds
    MAX_CONTROL_DURATION = 20  # Maximum: 20 steps = 1.0 second


# ============================================================================
# STATE PROPAGATOR
# ============================================================================

def propagate(start: Any, control: Any, duration: float, state: Any) -> None:
    """Propagate kinodynamic state forward in time using control inputs.
    
    This is the STATE PROPAGATOR function - the core of kinodynamic planning.
    OMPL calls this function to simulate how the robot moves when a control
    is applied. It implements the system dynamics (equations of motion).
    
    Args:
        start: Initial OMPL state (CompoundState with position and velocity subspaces)
               start[0] = position subspace (x, y)
               start[1] = velocity subspace (vx, vy)
        control: OMPL control (RealVectorControl with 2 components)
                 control[0] = acceleration in x direction (ax)
                 control[1] = acceleration in y direction (ay)
        duration: How long to apply the control (seconds)
        state: Output OMPL state where result is stored (modified in-place)
    
    Dynamics (Double Integrator Model):
    -----------------------------------
    The robot follows simple kinematic equations:
        dx/dt = vx           (x position changes based on x velocity)
        dy/dt = vy           (y position changes based on y velocity)
        dvx/dt = ax          (x velocity changes based on x acceleration - the control)
        dvy/dt = ay          (y velocity changes based on y acceleration - the control)
    
    Integration Method:
    -------------------
    Uses analytical solution with proper handling of velocity saturation:
    
    Case 1: Velocity stays within limits during entire time step
        - v_new = v_old + a * dt
        - x_new = x_old + v_old * dt + 0.5 * a * dt²
    
    Case 2: Velocity hits limit during time step
        - Calculate time t1 when velocity limit is reached: t1 = (v_limit - v_old) / a
        - Phase 1 (0 to t1): Accelerate until limit
          x1 = x_old + v_old * t1 + 0.5 * a * t1²
        - Phase 2 (t1 to dt): Continue at constant velocity limit
          x_new = x1 + v_limit * (dt - t1)
        - v_new = v_limit (clamped)
    
    This ensures position is physically consistent with velocity limits being enforced.
    """
    # -------- Extract Start State Components --------
    # OMPL CompoundState has multiple subspaces, access them by index
    pos_state = start[0]  # Position subspace (RealVectorStateSpace of dimension 2)
    vel_state = start[1]  # Velocity subspace (RealVectorStateSpace of dimension 2)
    
    # Get individual scalar values from each subspace
    x = pos_state[0]   # Current x position (meters)
    y = pos_state[1]   # Current y position (meters)
    vx = vel_state[0]  # Current x velocity (m/s)
    vy = vel_state[1]  # Current y velocity (m/s)
    
    # -------- Extract Control Inputs --------
    # Control is a RealVectorControl with 2 components (accelerations)
    ax = control[0]  # Acceleration command in x direction (m/s^2)
    ay = control[1]  # Acceleration command in y direction (m/s^2)
    
    # -------- Time Step --------
    dt = duration  # How long to apply this control (seconds)
    
    # -------- Integrate Dynamics: Update Velocities --------
    # Apply accelerations to compute new velocities: v_new = v_old + a * dt
    vx_unclamped = vx + ax * dt
    vy_unclamped = vy + ay * dt
    
    # -------- Enforce Velocity Limits and Compute Position --------
    # We need to handle cases where velocity limits are hit during the time step.
    # If velocity gets clamped, we calculate position in two phases:
    #   1. Accelerate until velocity limit is reached
    #   2. Continue at constant velocity limit for remaining time
    
    # Handle X direction
    if vx_unclamped > RobotConfig.MAX_VX:
        # Hit upper velocity limit
        vx_new = RobotConfig.MAX_VX
        # Time to reach velocity limit: t1 = (v_max - v0) / a
        if abs(ax) > 1e-10:  # Avoid division by zero
            t1 = (RobotConfig.MAX_VX - vx) / ax
            t1 = max(0.0, min(t1, dt))  # Clamp to [0, dt]
            # Position: accelerate for t1, then constant velocity for (dt - t1)
            x_new = x + vx * t1 + 0.5 * ax * t1 * t1 + RobotConfig.MAX_VX * (dt - t1)
        else:
            x_new = x + vx * dt  # No acceleration, use current velocity
    elif vx_unclamped < -RobotConfig.MAX_VX:
        # Hit lower velocity limit
        vx_new = -RobotConfig.MAX_VX
        if abs(ax) > 1e-10:
            t1 = (-RobotConfig.MAX_VX - vx) / ax
            t1 = max(0.0, min(t1, dt))
            x_new = x + vx * t1 + 0.5 * ax * t1 * t1 + (-RobotConfig.MAX_VX) * (dt - t1)
        else:
            x_new = x + vx * dt
    else:
        # No clamping needed - use analytical solution
        vx_new = vx_unclamped
        x_new = x + vx * dt + 0.5 * ax * dt * dt
    
    # Handle Y direction (same logic)
    if vy_unclamped > RobotConfig.MAX_VY:
        vy_new = RobotConfig.MAX_VY
        if abs(ay) > 1e-10:
            t1 = (RobotConfig.MAX_VY - vy) / ay
            t1 = max(0.0, min(t1, dt))
            y_new = y + vy * t1 + 0.5 * ay * t1 * t1 + RobotConfig.MAX_VY * (dt - t1)
        else:
            y_new = y + vy * dt
    elif vy_unclamped < -RobotConfig.MAX_VY:
        vy_new = -RobotConfig.MAX_VY
        if abs(ay) > 1e-10:
            t1 = (-RobotConfig.MAX_VY - vy) / ay
            t1 = max(0.0, min(t1, dt))
            y_new = y + vy * t1 + 0.5 * ay * t1 * t1 + (-RobotConfig.MAX_VY) * (dt - t1)
        else:
            y_new = y + vy * dt
    else:
        vy_new = vy_unclamped
        y_new = y + vy * dt + 0.5 * ay * dt * dt
    
    # -------- Enforce Position Limits --------
    # Clamp positions to workspace boundaries
    # This keeps the robot within the planning space
    x_new = np.clip(x_new, RobotConfig.MIN_X, RobotConfig.MAX_X)
    y_new = np.clip(y_new, RobotConfig.MIN_Y, RobotConfig.MAX_Y)
    
    # -------- Store Result in Output State --------
    # OMPL passes 'state' by reference, we modify it in-place
    result_pos = state[0]  # Get position subspace of result state
    result_vel = state[1]  # Get velocity subspace of result state
    
    # Set the computed values
    result_pos[0] = x_new    # Set new x position
    result_pos[1] = y_new    # Set new y position
    result_vel[0] = vx_new   # Set new x velocity
    result_vel[1] = vy_new   # Set new y velocity
    
    # Note: We don't return anything because 'state' is modified in-place
    # OMPL will use the updated 'state' as the result of propagation


# ============================================================================
# OBSTACLE CHECKER
# ============================================================================

class ObstacleChecker:
    """Obstacle checking for collision detection in 2D workspace.
    
    This class provides the STATE VALIDITY CHECKER for OMPL.
    OMPL calls is_valid_state() to determine if a state is collision-free.
    
    In this simple example, obstacles are represented as circles.
    You can extend this to support:
    - Rectangles
    - Polygons
    - Point clouds
    - Occupancy grids
    - Distance fields
    """
    
    def __init__(self) -> None:
        """Initialize obstacle checker with predefined obstacles.
        
        Obstacles are defined as circular regions in the 2D workspace.
        Each obstacle is a tuple: (center_x, center_y, radius)
        """
        # List of circular obstacles: (x_center, y_center, radius)
        self.obstacles: List[Tuple[float, float, float]] = [
            (2.0, 2.0, 1.0),    # Obstacle 1: center at (2, 2), radius 1.0m
            (5.0, -2.0, 1.2),   # Obstacle 2: center at (5, -2), radius 1.2m
            (-3.0, 4.0, 0.8),   # Obstacle 3: center at (-3, 4), radius 0.8m
        ]
    
    def is_valid_state(self, state: Any) -> bool:
        """Check if a state is collision-free (valid).
        
        This is the STATE VALIDITY CHECKER function that OMPL uses.
        It's called frequently during planning to check if sampled states
        and propagated trajectories are collision-free.
        
        Args:
            state: OMPL state (CompoundState with position and velocity)
                   We only need to check position for collision
        
        Returns:
            bool: True if state is valid (collision-free), False otherwise
        
        Note:
        -----
        For kinodynamic planning, we typically only check position for collisions,
        not velocity. Velocity is a "hidden" state that doesn't affect geometry.
        However, you could add velocity-based constraints here if needed
        (e.g., speed limits in certain zones).
        """
        # -------- Extract Position from State --------
        # State is a CompoundState: state[0] = position, state[1] = velocity
        pos = state[0]  # Get position subspace
        x = pos[0]      # Extract x coordinate
        y = pos[1]      # Extract y coordinate
        
        # -------- Check Collision with Each Obstacle --------
        for ox, oy, radius in self.obstacles:
            # Compute Euclidean distance from state position to obstacle center
            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
            
            # If distance is less than obstacle radius, we have a collision
            if dist < radius:
                return False  # State is INVALID (in collision)
        
        # If we made it here, no collisions detected
        return True  # State is VALID (collision-free)


# ============================================================================
# OMPL SETUP
# ============================================================================

def create_kinodynamic_space_information() -> Tuple[Any, ObstacleChecker]:
    """Create and configure OMPL space information for kinodynamic planning.
    
    This function sets up all the components OMPL needs for kinodynamic planning:
    1. State space (what is a state?)
    2. Control space (what control inputs are available?)
    3. State propagator (how do controls affect state?)
    4. State validity checker (which states are collision-free?)
    
    Returns:
        si: SpaceInformation object (contains all planning configuration)
        obstacle_checker: ObstacleChecker instance (for visualization later)
    
    OMPL Component Hierarchy:
    -------------------------
    StateSpace + ControlSpace → SpaceInformation → Planner
    
    The SpaceInformation (si) object is the central configuration that
    gets passed to planners. It knows:
    - What states look like
    - What controls are available
    - How to propagate states
    - Which states are valid
    """
    
    # ========================================================================
    # STEP 1: Create State Space (Define what a "state" is)
    # ========================================================================
    
    # For kinodynamic systems, state = (configuration, velocity)
    # We use a CompoundStateSpace to combine position and velocity
    # CompoundStateSpace: Allows combining multiple state space types
    state_space = ob.CompoundStateSpace()
    
    # -------- Position Subspace (x, y) --------
    # RealVectorStateSpace(n): State space of n-dimensional real vectors
    # This represents the robot's position in 2D workspace
    position_space = ob.RealVectorStateSpace(2)  # 2D position: (x, y)
    
    # Set bounds on position (workspace limits)
    # RealVectorBounds: Defines min/max values for each dimension
    pos_bounds = ob.RealVectorBounds(2)
    pos_bounds.setLow(0, RobotConfig.MIN_X)   # Set x minimum
    pos_bounds.setHigh(0, RobotConfig.MAX_X)  # Set x maximum
    pos_bounds.setLow(1, RobotConfig.MIN_Y)   # Set y minimum
    pos_bounds.setHigh(1, RobotConfig.MAX_Y)  # Set y maximum
    position_space.setBounds(pos_bounds)
    
    # -------- Velocity Subspace (vx, vy) --------
    # Similar to position, but represents velocities
    velocity_space = ob.RealVectorStateSpace(2)  # 2D velocity: (vx, vy)
    
    # Set bounds on velocity (physical limits)
    vel_bounds = ob.RealVectorBounds(2)
    vel_bounds.setLow(0, -RobotConfig.MAX_VX)   # vx minimum (negative = left)
    vel_bounds.setHigh(0, RobotConfig.MAX_VX)   # vx maximum (positive = right)
    vel_bounds.setLow(1, -RobotConfig.MAX_VY)   # vy minimum (negative = down)
    vel_bounds.setHigh(1, RobotConfig.MAX_VY)   # vy maximum (positive = up)
    velocity_space.setBounds(vel_bounds)
    
    # -------- Combine Position and Velocity into Compound State --------
    # addSubspace(subspace, weight): Add a subspace with importance weight
    # Weights affect distance metrics (how "far apart" states are)
    # Higher weight = more important when computing distances
    state_space.addSubspace(position_space, 1.0)  # Position weight = 1.0
    state_space.addSubspace(velocity_space, 0.3)  # Velocity weight = 0.3
    # Result: Position differences matter more than velocity differences
    
    # ========================================================================
    # STEP 2: Create Control Space (Define available control inputs)
    # ========================================================================
    
    # RealVectorControlSpace(state_space, dim): Control space of dim-dimensional vectors
    # Takes state_space as argument to associate controls with states
    # Our controls are accelerations: (ax, ay)
    control_space = oc.RealVectorControlSpace(state_space, 2)  # 2D control
    
    # Set bounds on control inputs (acceleration limits)
    control_bounds = ob.RealVectorBounds(2)
    control_bounds.setLow(0, -RobotConfig.MAX_AX)   # ax minimum (max deceleration)
    control_bounds.setHigh(0, RobotConfig.MAX_AX)   # ax maximum (max acceleration)
    control_bounds.setLow(1, -RobotConfig.MAX_AY)   # ay minimum
    control_bounds.setHigh(1, RobotConfig.MAX_AY)   # ay maximum
    control_space.setBounds(control_bounds)
    
    # ========================================================================
    # STEP 3: Create Space Information (Combine everything)
    # ========================================================================
    
    # SpaceInformation: Central object that holds all planning configuration
    # For kinodynamic planning, we use oc.SpaceInformation (control variant)
    # This differs from ob.SpaceInformation used in geometric planning
    si = oc.SpaceInformation(state_space, control_space)
    
    # -------- Configure State Propagator --------
    # The propagator defines system dynamics (how controls affect state)
    # StatePropagatorFn: Wraps a Python function to use as propagator
    # OMPL will call our propagate() function during planning
    si.setStatePropagator(oc.StatePropagatorFn(propagate))
    
    # setPropagationStepSize: Time step for numerical integration
    # This is the 'dt' passed to propagate() function
    # Smaller = more accurate but slower; Larger = faster but less accurate
    si.setPropagationStepSize(RobotConfig.PROPAGATION_STEP_SIZE)
    
    # setMinMaxControlDuration: Range of how long controls are applied
    # Arguments are in "number of propagation steps", not seconds
    # Actual time = (num_steps * propagation_step_size)
    # OMPL will sample random durations in this range
    si.setMinMaxControlDuration(
        RobotConfig.MIN_CONTROL_DURATION,  # Minimum duration (steps)
        RobotConfig.MAX_CONTROL_DURATION   # Maximum duration (steps)
    )
    
    # -------- Configure State Validity Checker --------
    # The validity checker determines if states are collision-free
    # StateValidityCheckerFn: Wraps a Python function as validity checker
    obstacle_checker = ObstacleChecker()
    si.setStateValidityChecker(ob.StateValidityCheckerFn(obstacle_checker.is_valid_state))
    
    # -------- Finalize Setup --------
    # setup(): Performs internal initialization and validation
    # Must be called before using SpaceInformation
    si.setup()
    
    # Return both si and obstacle_checker (latter needed for visualization)
    return si, obstacle_checker


def create_state(si: Any, x: float, y: float, vx: float, vy: float) -> Any:
    """Helper function to create and initialize an OMPL state.
    
    This convenience function allocates a new state and sets its values.
    OMPL requires proper allocation through SpaceInformation.
    
    Args:
        si: SpaceInformation object (knows how to allocate states)
        x, y: Position coordinates (meters)
        vx, vy: Velocity components (m/s)
    
    Returns:
        OMPL state with specified position and velocity
    
    Note:
    -----
    We must use si.allocState() rather than creating states manually.
    This ensures proper memory management within OMPL.
    """
    # allocState(): Allocates a new state with proper memory management
    # The returned state has the structure defined in our CompoundStateSpace
    state = si.allocState()
    
    # Set position components (first subspace, index 0)
    state[0][0] = x   # x position
    state[0][1] = y   # y position
    
    # Set velocity components (second subspace, index 1)
    state[1][0] = vx  # x velocity
    state[1][1] = vy  # y velocity
    
    return state


def plan_kinodynamic_rrt_star(
    start_pos: Tuple[float, float], 
    goal_pos: Tuple[float, float], 
    start_vel: Tuple[float, float] = (0.0, 0.0), 
    goal_vel: Tuple[float, float] = (0.0, 0.0), 
    planning_time: float = 10.0
) -> Tuple[Optional[List[Tuple[float, float, float, float]]], 
           Optional[List[Tuple[float, float, float]]], 
           Dict[str, Any]]:
    """Plan a kinodynamic trajectory using RRT algorithm.
    
    This is the main planning function that sets up and runs OMPL's
    RRT (Rapidly-exploring Random Tree) planner for kinodynamic systems.
    
    RRT Algorithm Overview:
    -----------------------
    1. Start with a tree containing just the start state
    2. Repeat until goal found or time limit:
       a. Sample a random state in the state space
       b. Find nearest node in tree to random state
       c. Sample a random control input
       d. Apply control for random duration (propagate state)
       e. If resulting state is valid (collision-free), add to tree
    3. Once goal region reached, extract path from start to goal
    
    Args:
        start_pos: Tuple (x, y) - starting position in meters
        goal_pos: Tuple (x, y) - goal position in meters
        start_vel: Tuple (vx, vy) - starting velocity in m/s (default: stationary)
        goal_vel: Tuple (vx, vy) - goal velocity in m/s (default: stationary)
        planning_time: Maximum time to spend planning in seconds (default: 10s)
    
    Returns:
        solution_path: List of states [(x, y, vx, vy), ...] along the trajectory
        solution_controls: List of controls [(ax, ay, duration), ...] to follow path
        planning_stats: Dictionary with planning statistics
            - 'solved': bool, whether solution was found
            - 'planning_time': float, actual time spent planning
            - 'path_length': float, geometric length of path in meters
            - 'num_states': int, number of states in solution path
            - 'duration': float, total time duration of trajectory in seconds
    
    Note on RRT vs RRT*:
    -------------------
    This implementation uses basic RRT. To use RRT* (asymptotically optimal):
    - Replace oc.RRT with oc.KPIECE1 or implement RRT* for control systems
    - RRT* performs rewiring to optimize paths, but is more complex for kinodynamic systems
    - For kinodynamic planning, SST (Stable Sparse RRT) is often preferred over RRT*
    """
    
    print("=" * 70)
    print("OMPL KINODYNAMIC RRT PLANNING")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: Create Space Information (state space, control space, dynamics)
    # ========================================================================
    
    # This sets up all the OMPL infrastructure: state space, control space,
    # propagation function, validity checker, etc.
    si, obstacle_checker = create_kinodynamic_space_information()
    
    # ========================================================================
    # STEP 2: Create Start and Goal States
    # ========================================================================
    
    # Allocate and initialize start state from user-provided position/velocity
    start_state = create_state(si, start_pos[0], start_pos[1], start_vel[0], start_vel[1])
    
    # Allocate and initialize goal state from user-provided position/velocity
    goal_state = create_state(si, goal_pos[0], goal_pos[1], goal_vel[0], goal_vel[1])
    
    print(f"Start: pos=({start_pos[0]:.2f}, {start_pos[1]:.2f}), "
          f"vel=({start_vel[0]:.2f}, {start_vel[1]:.2f})")
    print(f"Goal:  pos=({goal_pos[0]:.2f}, {goal_pos[1]:.2f}), "
          f"vel=({goal_vel[0]:.2f}, {goal_vel[1]:.2f})")
    print()
    
    # ========================================================================
    # STEP 3: Create Problem Definition
    # ========================================================================
    
    # ProblemDefinition: Specifies the planning problem
    # Takes SpaceInformation to know about state/control spaces
    pdef = ob.ProblemDefinition(si)
    
    # setStartAndGoalStates: Define start, goal, and goal tolerance
    # Arguments:
    #   - start: Initial state where robot starts
    #   - goal: Target state we want to reach
    #   - threshold: How close we need to get to goal (in state space distance)
    #                Smaller threshold = more precise but potentially slower
    #                For our compound space: threshold includes both position and velocity
    pdef.setStartAndGoalStates(start_state, goal_state, threshold=0.5)
    
    # Alternative: Use setGoalState() with a GoalState object for custom goal regions
    # This is useful for goal regions that aren't just a threshold around a point
    
    # ========================================================================
    # STEP 4: Create and Configure Planner
    # ========================================================================
    
    # Choose planner: oc.RRT for basic kinodynamic planning
    # Other options demonstrate different exploration strategies:
    #   - oc.KPIECE1: Cell decomposition, more structured exploration
    #   - oc.EST: Expansive Space Trees, good for high-dimensional spaces
    #   - oc.SST: Stable Sparse RRT, asymptotically optimal for kinodynamic
    # 
    # Note: True steering functions would use DirectedControlSampler, but that's
    # complex in Python bindings. These planners use control sampling but with
    # different strategies for how they explore the space.
    planner = oc.RRT(si)
    
    # Optional: Set goal bias for RRT (makes it more "steering-like")
    # planner.setGoalBias(0.1)  # 10% chance of sampling toward goal
    
    # setProblemDefinition: Tell planner what problem to solve
    planner.setProblemDefinition(pdef)
    
    # setup(): Initialize planner internal data structures
    # Must be called before solve()
    planner.setup()
    
    # ========================================================================
    # STEP 5: Solve the Planning Problem
    # ========================================================================
    
    print(f"Planning for {planning_time:.1f} seconds...")
    start_time = time.time()
    
    # solve(time): Run planner for specified time limit (seconds)
    # Returns PlannerStatus indicating success/failure
    # Possible return values:
    #   - EXACT_SOLUTION: Found path that satisfies goal exactly
    #   - APPROXIMATE_SOLUTION: Found path close to goal (within threshold)
    #   - TIMEOUT: Ran out of time without finding solution
    #   - INVALID_START: Start state is invalid (e.g., in collision)
    #   - INVALID_GOAL: Goal state is invalid
    solved = planner.solve(planning_time)
    
    elapsed_time = time.time() - start_time
    
    # ========================================================================
    # STEP 6: Extract Solution (if found)
    # ========================================================================
    
    # Initialize statistics dictionary
    stats = {
        'solved': False,
        'planning_time': elapsed_time,
        'path_length': 0,
        'num_states': 0,
        'duration': 0.0
    }
    
    if solved:
        print(f"✓ Solution found in {elapsed_time:.2f}s!")
        
        # ====================================================================
        # Extract Solution Path from Problem Definition
        # ====================================================================
        
        # getSolutionPath(): Returns the path found by the planner
        # For kinodynamic planning, this is a control::PathControl object
        # PathControl contains:
        #   - Sequence of states (waypoints along trajectory)
        #   - Sequence of controls (what to do between waypoints)
        #   - Duration for each control (how long to apply it)
        path = pdef.getSolutionPath()
        
        # ====================================================================
        # Extract States and Controls from PathControl Object
        # ====================================================================
        
        # PathControl is a sequence: state0 --(control0, duration0)--> state1 --(control1, duration1)--> ...
        # We'll extract these into Python lists for easier processing
        
        solution_path = []      # List of (x, y, vx, vy) tuples
        solution_controls = []  # List of (ax, ay, duration) tuples
        
        # getStateCount(): Returns number of states in the path
        # This includes the start state, so num_states >= 1
        num_states = path.getStateCount()
        stats['num_states'] = num_states
        
        # -------- Iterate Through Path States --------
        for i in range(num_states):
            # getState(i): Get the i-th state along the path
            # Returns an OMPL State object (CompoundState in our case)
            state = path.getState(i)
            
            # Extract position and velocity from compound state
            x = state[0][0]   # x position from position subspace
            y = state[0][1]   # y position from position subspace
            vx = state[1][0]  # x velocity from velocity subspace
            vy = state[1][1]  # y velocity from velocity subspace
            
            # Store as tuple
            solution_path.append((x, y, vx, vy))
            
            # -------- Extract Control (if not the last state) --------
            # There are N-1 controls for N states (no control after last state)
            if i < num_states - 1:
                # getControl(i): Get control that transitions from state i to state i+1
                # Returns an OMPL Control object (RealVectorControl in our case)
                control = path.getControl(i)
                
                # getControlDuration(i): Get how long control i is applied (seconds)
                # This is the actual simulation time for this segment
                duration = path.getControlDuration(i)
                
                # Extract acceleration components from control
                ax = control[0]  # Acceleration in x direction
                ay = control[1]  # Acceleration in y direction
                
                # Store as tuple
                solution_controls.append((ax, ay, duration))
                
                # Accumulate total trajectory duration
                stats['duration'] += duration
        
        # ====================================================================
        # Calculate Path Statistics
        # ====================================================================
        
        # ====================================================================
        # Calculate Path Statistics
        # ====================================================================
        
        # Calculate geometric path length (sum of Euclidean distances)
        # This measures how far the robot travels in space
        path_length = 0.0
        for i in range(len(solution_path) - 1):
            # Extract consecutive state positions
            x1, y1, _, _ = solution_path[i]
            x2, y2, _, _ = solution_path[i + 1]
            
            # Add Euclidean distance between positions
            path_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Update statistics
        stats['solved'] = True
        stats['path_length'] = path_length
        
        # Print summary
        print(f"Path length: {path_length:.2f} m")
        print(f"Duration: {stats['duration']:.2f} s")
        print(f"Number of states: {num_states}")
        print()
        
        return solution_path, solution_controls, stats
    else:
        # Planning failed - no solution found within time limit
        print("✗ No solution found within time limit.")
        print()
        return None, None, stats


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_solution(
    solution_path: Optional[List[Tuple[float, float, float, float]]], 
    solution_controls: Optional[List[Tuple[float, float, float]]], 
    start_pos: Tuple[float, float], 
    goal_pos: Tuple[float, float], 
    obstacle_checker: ObstacleChecker, 
    stats: Dict[str, Any]
) -> None:
    """Visualize the planned kinodynamic trajectory with matplotlib.
    
    Creates a two-panel figure:
    - Left: 2D workspace showing trajectory, obstacles, and velocity vectors
    - Right: Time profiles of velocities and accelerations
    
    Args:
        solution_path: List of states [(x, y, vx, vy), ...]
        solution_controls: List of controls [(ax, ay, duration), ...]
        start_pos: Tuple (x, y) - starting position
        goal_pos: Tuple (x, y) - goal position
        obstacle_checker: ObstacleChecker instance (for drawing obstacles)
        stats: Dictionary with planning statistics
    """
    
    # Check if there's a solution to visualize
    if solution_path is None:
        print("No solution to visualize.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # -------- Plot 1: Trajectory in workspace --------
    ax1.set_xlim(RobotConfig.MIN_X, RobotConfig.MAX_X)
    ax1.set_ylim(RobotConfig.MIN_Y, RobotConfig.MAX_Y)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Kinodynamic RRT* Trajectory', fontsize=14, fontweight='bold')
    
    # Draw obstacles
    for ox, oy, radius in obstacle_checker.obstacles:
        circle = Circle((ox, oy), radius, color='red', alpha=0.5, label='Obstacle')
        ax1.add_patch(circle)
    
    # Extract positions
    x_coords = [s[0] for s in solution_path]
    y_coords = [s[1] for s in solution_path]
    
    # Plot trajectory
    ax1.plot(x_coords, y_coords, 'b-', linewidth=2, label='Trajectory', alpha=0.7)
    ax1.plot(x_coords, y_coords, 'b.', markersize=4, alpha=0.5)
    
    # Mark start and goal
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=15, label='Start', 
             markeredgecolor='black', markeredgewidth=2)
    ax1.plot(goal_pos[0], goal_pos[1], 'r*', markersize=20, label='Goal',
             markeredgecolor='black', markeredgewidth=1.5)
    
    # Add velocity arrows at some points
    arrow_step = max(1, len(solution_path) // 15)
    for i in range(0, len(solution_path), arrow_step):
        x, y, vx, vy = solution_path[i]
        # Scale arrow for visibility
        scale = 0.3
        ax1.arrow(x, y, vx * scale, vy * scale, 
                 head_width=0.2, head_length=0.15, 
                 fc='green', ec='darkgreen', alpha=0.6, linewidth=1.5)
    
    ax1.legend(loc='upper left', fontsize=10)
    
    # Add stats text
    stats_text = f"Planning Time: {stats['planning_time']:.2f}s\n"
    stats_text += f"Path Length: {stats['path_length']:.2f}m\n"
    stats_text += f"Duration: {stats['duration']:.2f}s\n"
    stats_text += f"States: {stats['num_states']}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # -------- Plot 2: Velocity and Acceleration profiles --------
    time_points = [0.0]
    vx_profile = [solution_path[0][2]]
    vy_profile = [solution_path[0][3]]
    ax_profile = []
    ay_profile = []
    
    current_time = 0.0
    for i, (ax, ay, duration) in enumerate(solution_controls):
        current_time += duration
        time_points.append(current_time)
        vx_profile.append(solution_path[i + 1][2])
        vy_profile.append(solution_path[i + 1][3])
        ax_profile.extend([ax, ax])
        ay_profile.extend([ay, ay])
    
    # Create time array for accelerations (piecewise constant)
    accel_times = []
    t = 0.0
    for i, (_, _, duration) in enumerate(solution_controls):
        accel_times.extend([t, t + duration])
        t += duration
    
    # Velocity subplot
    ax2_vel = ax2
    ax2_vel.plot(time_points, vx_profile, 'b-', linewidth=2, label='vx')
    ax2_vel.plot(time_points, vy_profile, 'r-', linewidth=2, label='vy')
    ax2_vel.axhline(y=RobotConfig.MAX_VX, color='b', linestyle='--', 
                    alpha=0.5, label='vx limit')
    ax2_vel.axhline(y=-RobotConfig.MAX_VX, color='b', linestyle='--', alpha=0.5)
    ax2_vel.axhline(y=RobotConfig.MAX_VY, color='r', linestyle='--', 
                    alpha=0.5, label='vy limit')
    ax2_vel.axhline(y=-RobotConfig.MAX_VY, color='r', linestyle='--', alpha=0.5)
    ax2_vel.set_xlabel('Time (s)', fontsize=12)
    ax2_vel.set_ylabel('Velocity (m/s)', fontsize=12)
    ax2_vel.set_title('Velocity Profile', fontsize=12, fontweight='bold')
    ax2_vel.legend(loc='upper right', fontsize=10)
    ax2_vel.grid(True, alpha=0.3)
    
    # Acceleration subplot (create second y-axis)
    ax2_accel = ax2_vel.twinx()
    if len(accel_times) > 0:
        ax2_accel.plot(accel_times, ax_profile, 'g-', linewidth=1.5, 
                      alpha=0.6, label='ax')
        ax2_accel.plot(accel_times, ay_profile, 'm-', linewidth=1.5, 
                      alpha=0.6, label='ay')
    ax2_accel.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax2_accel.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('kinodynamic_rrt_star_result.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'kinodynamic_rrt_star_result.png'")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run kinodynamic RRT planning example with visualization.
    
    This is the main entry point that:
    1. Defines a planning problem (start, goal, obstacles)
    2. Calls the OMPL planner
    3. Visualizes the result if successful
    
    Modify the parameters below to test different scenarios.
    """
    
    # ========================================================================
    # Define Planning Problem
    # ========================================================================
    
    # Starting configuration
    start_pos = (-8.0, -8.0)  # Start in bottom-left corner
    start_vel = (0.0, 0.0)    # Start from rest (stationary)
    
    # Goal configuration
    goal_pos = (8.0, 7.0)     # Goal in top-right area
    goal_vel = (0.0, 0.0)     # End at rest (stationary)
    
    # Planning time budget
    planning_time = 10.0      # Maximum 10 seconds of planning
    
    # Note: Obstacles are defined in ObstacleChecker class above
    # Modify that class to change obstacle locations/sizes
    
    # ========================================================================
    # Run Planner
    # ========================================================================
    
    solution_path, solution_controls, stats = plan_kinodynamic_rrt_star(
        start_pos=start_pos,
        goal_pos=goal_pos,
        start_vel=start_vel,
        goal_vel=goal_vel,
        planning_time=planning_time
    )
    
    # ========================================================================
    # Visualize Result
    # ========================================================================
    
    if stats['solved']:
        # Planning succeeded - create visualization
        _, obstacle_checker = create_kinodynamic_space_information()
        visualize_solution(solution_path, solution_controls, 
                          start_pos, goal_pos, obstacle_checker, stats)
    else:
        # Planning failed - provide suggestions
        print("Planning failed. Suggestions:")
        print("  - Increase planning_time parameter")
        print("  - Relax goal threshold in plan_kinodynamic_rrt_star()")
        print("  - Check if start/goal are in collision")
        print("  - Adjust velocity/acceleration limits if too restrictive")
    
    return stats['solved']


if __name__ == "__main__":
    """
    Main execution block.
    
    This script demonstrates OMPL's kinodynamic planning capabilities.
    
    What happens when you run this script:
    1. OMPL sets up state space (position + velocity)
    2. OMPL sets up control space (accelerations)
    3. RRT algorithm builds a tree of feasible trajectories
    4. Solution is extracted and visualized
    
    Key Takeaways:
    - State propagation enforces dynamics (can't teleport, must obey physics)
    - Controls must respect acceleration limits
    - Velocities are automatically limited during propagation
    - Result is a dynamically feasible trajectory
    
    To Experiment:
    - Change obstacle positions/sizes in ObstacleChecker.__init__()
    - Modify limits in RobotConfig class
    - Try different start/goal configurations in main()
    - Switch planner: Replace oc.RRT with oc.KPIECE1, oc.EST, etc.
    - Adjust goal threshold for more/less precision
    """
    
    print("\n" + "=" * 70)
    print("OMPL Kinodynamic RRT 2D Planning")
    print("=" * 70 + "\n")
    
    success = main()
    
    print("\n" + "=" * 70)
    if success:
        print("✓ Planning completed successfully!")
    else:
        print("✗ Planning failed.")
    print("=" * 70 + "\n")

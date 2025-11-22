#!/usr/bin/env python3
"""
OMPL Kinodynamic SST (RRT*-like) for 6DoF Double Integrator System
Moved to six_dof directory during repository reorganization.
Original functionality preserved.
"""
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from ompl import base as ob
from ompl import control as oc
import time

class RobotConfig6DoF:
    MIN_X, MAX_X = -10.0, 10.0
    MIN_Y, MAX_Y = -10.0, 10.0
    MIN_Z, MAX_Z = -5.0, 5.0
    MIN_ROLL, MAX_ROLL = -np.pi, np.pi
    MIN_PITCH, MAX_PITCH = -np.pi/2, np.pi/2
    MIN_YAW, MAX_YAW = -np.pi, np.pi
    MAX_VX, MAX_VY, MAX_VZ = 2.0, 2.0, 1.5
    MAX_VROLL, MAX_VPITCH, MAX_VYAW = 1.5, 1.5, 2.0
    MAX_AX, MAX_AY, MAX_AZ = 3.0, 3.0, 2.5
    MAX_AROLL, MAX_APITCH, MAX_AYAW = 2.0, 2.0, 2.5
    PROPAGATION_STEP_SIZE = 0.05
    MIN_CONTROL_DURATION, MAX_CONTROL_DURATION = 5, 20

class ObstacleChecker6DoF:
    def __init__(self) -> None:
        self.obstacles: List[Tuple[float, float, float, float]] = [
            (2.0, 2.0, 0.0, 1.2), (5.0, -2.0, 1.0, 1.5), (-3.0, 4.0, -1.0, 1.0), (0.0, 0.0, 2.0, 0.8), (-5.0, -5.0, 0.5, 1.3)
        ]
    def is_valid_state(self, state: Any) -> bool:
        x, y, z = state[0][0], state[0][1], state[0][2]
        for ox, oy, oz, radius in self.obstacles:
            if np.sqrt((x - ox)**2 + (y - oy)**2 + (z - oz)**2) < radius:
                return False
        return True

def propagate_6dof(start: Any, control: Any, duration: float, state: Any) -> None:
    x, y, z = start[0][0], start[0][1], start[0][2]
    roll, pitch, yaw = start[0][3], start[0][4], start[0][5]
    vx, vy, vz = start[1][0], start[1][1], start[1][2]
    vroll, vpitch, vyaw = start[1][3], start[1][4], start[1][5]
    ax, ay, az, aroll, apitch, ayaw = control[0], control[1], control[2], control[3], control[4], control[5]
    dt = duration
    def propagate_1d(pos: float, vel: float, acc: float, max_vel: float, min_pos: float, max_pos: float) -> Tuple[float, float]:
        vel_unclamped = vel + acc * dt
        if vel_unclamped > max_vel:
            vel_new = max_vel
            if abs(acc) > 1e-10:
                t1 = max(0.0, min((max_vel - vel) / acc, dt))
                pos_new = pos + vel * t1 + 0.5 * acc * t1**2 + max_vel * (dt - t1)
            else:
                pos_new = pos + vel * dt
        elif vel_unclamped < -max_vel:
            vel_new = -max_vel
            if abs(acc) > 1e-10:
                t1 = max(0.0, min((-max_vel - vel) / acc, dt))
                pos_new = pos + vel * t1 + 0.5 * acc * t1**2 - max_vel * (dt - t1)
            else:
                pos_new = pos + vel * dt
        else:
            vel_new = vel_unclamped
            pos_new = pos + vel * dt + 0.5 * acc * dt**2
        pos_new = np.clip(pos_new, min_pos, max_pos)
        return pos_new, vel_new
    x_new, vx_new = propagate_1d(x, vx, ax, RobotConfig6DoF.MAX_VX, RobotConfig6DoF.MIN_X, RobotConfig6DoF.MAX_X)
    y_new, vy_new = propagate_1d(y, vy, ay, RobotConfig6DoF.MAX_VY, RobotConfig6DoF.MIN_Y, RobotConfig6DoF.MAX_Y)
    z_new, vz_new = propagate_1d(z, vz, az, RobotConfig6DoF.MAX_VZ, RobotConfig6DoF.MIN_Z, RobotConfig6DoF.MAX_Z)
    roll_new, vroll_new = propagate_1d(roll, vroll, aroll, RobotConfig6DoF.MAX_VROLL, RobotConfig6DoF.MIN_ROLL, RobotConfig6DoF.MAX_ROLL)
    pitch_new, vpitch_new = propagate_1d(pitch, vpitch, apitch, RobotConfig6DoF.MAX_VPITCH, RobotConfig6DoF.MIN_PITCH, RobotConfig6DoF.MAX_PITCH)
    yaw_new, vyaw_new = propagate_1d(yaw, vyaw, ayaw, RobotConfig6DoF.MAX_VYAW, RobotConfig6DoF.MIN_YAW, RobotConfig6DoF.MAX_YAW)
    roll_new = np.arctan2(np.sin(roll_new), np.cos(roll_new))
    pitch_new = np.arctan2(np.sin(pitch_new), np.cos(pitch_new))
    yaw_new = np.arctan2(np.sin(yaw_new), np.cos(yaw_new))
    state[0][0], state[0][1], state[0][2] = x_new, y_new, z_new
    state[0][3], state[0][4], state[0][5] = roll_new, pitch_new, yaw_new
    state[1][0], state[1][1], state[1][2] = vx_new, vy_new, vz_new
    state[1][3], state[1][4], state[1][5] = vroll_new, vpitch_new, vyaw_new

def create_6dof_space_information() -> Tuple[Any, ObstacleChecker6DoF]:
    state_space = ob.CompoundStateSpace()
    pos_space = ob.RealVectorStateSpace(6)
    pos_bounds = ob.RealVectorBounds(6)
    pos_bounds.setLow(0, RobotConfig6DoF.MIN_X); pos_bounds.setHigh(0, RobotConfig6DoF.MAX_X)
    pos_bounds.setLow(1, RobotConfig6DoF.MIN_Y); pos_bounds.setHigh(1, RobotConfig6DoF.MAX_Y)
    pos_bounds.setLow(2, RobotConfig6DoF.MIN_Z); pos_bounds.setHigh(2, RobotConfig6DoF.MAX_Z)
    pos_bounds.setLow(3, RobotConfig6DoF.MIN_ROLL); pos_bounds.setHigh(3, RobotConfig6DoF.MAX_ROLL)
    pos_bounds.setLow(4, RobotConfig6DoF.MIN_PITCH); pos_bounds.setHigh(4, RobotConfig6DoF.MAX_PITCH)
    pos_bounds.setLow(5, RobotConfig6DoF.MIN_YAW); pos_bounds.setHigh(5, RobotConfig6DoF.MAX_YAW)
    pos_space.setBounds(pos_bounds)
    vel_space = ob.RealVectorStateSpace(6)
    vel_bounds = ob.RealVectorBounds(6)
    vel_bounds.setLow(0, -RobotConfig6DoF.MAX_VX); vel_bounds.setHigh(0, RobotConfig6DoF.MAX_VX)
    vel_bounds.setLow(1, -RobotConfig6DoF.MAX_VY); vel_bounds.setHigh(1, RobotConfig6DoF.MAX_VY)
    vel_bounds.setLow(2, -RobotConfig6DoF.MAX_VZ); vel_bounds.setHigh(2, RobotConfig6DoF.MAX_VZ)
    vel_bounds.setLow(3, -RobotConfig6DoF.MAX_VROLL); vel_bounds.setHigh(3, RobotConfig6DoF.MAX_VROLL)
    vel_bounds.setLow(4, -RobotConfig6DoF.MAX_VPITCH); vel_bounds.setHigh(4, RobotConfig6DoF.MAX_VPITCH)
    vel_bounds.setLow(5, -RobotConfig6DoF.MAX_VYAW); vel_bounds.setHigh(5, RobotConfig6DoF.MAX_VYAW)
    vel_space.setBounds(vel_bounds)
    state_space.addSubspace(pos_space, 1.0)
    state_space.addSubspace(vel_space, 0.3)
    control_space = oc.RealVectorControlSpace(state_space, 6)
    control_bounds = ob.RealVectorBounds(6)
    control_bounds.setLow(0, -RobotConfig6DoF.MAX_AX); control_bounds.setHigh(0, RobotConfig6DoF.MAX_AX)
    control_bounds.setLow(1, -RobotConfig6DoF.MAX_AY); control_bounds.setHigh(1, RobotConfig6DoF.MAX_AY)
    control_bounds.setLow(2, -RobotConfig6DoF.MAX_AZ); control_bounds.setHigh(2, RobotConfig6DoF.MAX_AZ)
    control_bounds.setLow(3, -RobotConfig6DoF.MAX_AROLL); control_bounds.setHigh(3, RobotConfig6DoF.MAX_AROLL)
    control_bounds.setLow(4, -RobotConfig6DoF.MAX_APITCH); control_bounds.setHigh(4, RobotConfig6DoF.MAX_APITCH)
    control_bounds.setLow(5, -RobotConfig6DoF.MAX_AYAW); control_bounds.setHigh(5, RobotConfig6DoF.MAX_AYAW)
    control_space.setBounds(control_bounds)
    si = oc.SpaceInformation(state_space, control_space)
    obstacle_checker = ObstacleChecker6DoF()
    si.setStateValidityChecker(ob.StateValidityCheckerFn(obstacle_checker.is_valid_state))
    si.setStatePropagator(oc.StatePropagatorFn(propagate_6dof))
    si.setPropagationStepSize(RobotConfig6DoF.PROPAGATION_STEP_SIZE)
    si.setMinMaxControlDuration(RobotConfig6DoF.MIN_CONTROL_DURATION, RobotConfig6DoF.MAX_CONTROL_DURATION)
    si.setup()
    return si, obstacle_checker

def create_6dof_state(si: Any, x: float, y: float, z: float, roll: float, pitch: float, yaw: float,
                      vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
                      vroll: float = 0.0, vpitch: float = 0.0, vyaw: float = 0.0) -> Any:
    state = si.allocState()
    state[0][0], state[0][1], state[0][2] = x, y, z
    state[0][3], state[0][4], state[0][5] = roll, pitch, yaw
    state[1][0], state[1][1], state[1][2] = vx, vy, vz
    state[1][3], state[1][4], state[1][5] = vroll, vpitch, vyaw
    return state

def compute_path_metrics(path: oc.PathControl) -> Tuple[float, float, float, int]:
    linear_length = 0.0
    angular_distance = 0.0
    duration = 0.0
    num_states: int = path.getStateCount()
    for i in range(num_states - 1):
        s1 = path.getState(i); s2 = path.getState(i+1)
        dx = s2[0][0] - s1[0][0]; dy = s2[0][1] - s1[0][1]; dz = s2[0][2] - s1[0][2]
        linear_length += np.sqrt(dx**2 + dy**2 + dz**2)
        droll = abs(s2[0][3] - s1[0][3]); dpitch = abs(s2[0][4] - s1[0][4]); dyaw = abs(s2[0][5] - s1[0][5])
        droll = min(droll, 2*np.pi - droll); dpitch = min(dpitch, 2*np.pi - dpitch); dyaw = min(dyaw, 2*np.pi - dyaw)
        angular_distance += np.sqrt(droll**2 + dpitch**2 + dyaw**2)
    for i in range(path.getControlCount()):
        duration += path.getControlDuration(i)
    return linear_length, angular_distance, duration, num_states

def plan_6dof_kinodynamic_rrt(start_pos: Tuple[float, float, float], start_orient: Tuple[float, float, float],
                              goal_pos: Tuple[float, float, float], goal_orient: Tuple[float, float, float],
                              start_vel: Tuple[float, float, float, float, float, float] = (0.0,0.0,0.0,0.0,0.0,0.0),
                              goal_vel: Tuple[float, float, float, float, float, float] = (0.0,0.0,0.0,0.0,0.0,0.0),
                              planning_time: float = 30.0) -> Dict[str, Any]:
    si, _ = create_6dof_space_information()
    start_state = create_6dof_state(si, start_pos[0], start_pos[1], start_pos[2], start_orient[0], start_orient[1], start_orient[2], *start_vel)
    goal_state = create_6dof_state(si, goal_pos[0], goal_pos[1], goal_pos[2], goal_orient[0], goal_orient[1], goal_orient[2], *goal_vel)
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state, 0.5)
    planner = oc.SST(si)
    planner.setProblemDefinition(pdef)
    planner.setSelectionRadius(0.2)
    planner.setPruningRadius(0.1)
    planner.setup()
    start_time = time.time()
    solved = planner.solve(planning_time)
    elapsed = time.time() - start_time
    stats = {'solved': False, 'planning_time': elapsed, 'linear_length': 0.0, 'angular_distance': 0.0, 'duration': 0.0, 'num_states': 0}
    if solved and pdef.hasSolution():
        path = pdef.getSolutionPath()
        linear_length, angular_distance, duration, num_states = compute_path_metrics(path)
        stats.update({'solved': True, 'linear_length': linear_length, 'angular_distance': angular_distance, 'duration': duration, 'num_states': num_states})
    return stats

if __name__ == '__main__':
    print("6DoF SST planner demo (six_dof)")
    stats = plan_6dof_kinodynamic_rrt((-8.0,-8.0,-3.0),(0.0,0.0,0.0),(8.0,7.0,2.0),(0.0,0.0,0.0), planning_time=10.0)
    print(stats)

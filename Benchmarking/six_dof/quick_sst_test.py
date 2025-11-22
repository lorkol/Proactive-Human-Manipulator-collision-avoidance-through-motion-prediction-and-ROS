#!/usr/bin/env python3
"""Quick SST test (moved to six_dof)."""
import sys
ROOT_PATH = '/home/ethan/ros2_ws/src/Proactive-Human-Manipulator-collision-avoidance-through-motion-prediction-and-ROS/Benchmarking'
SIX_DOF_PATH = f'{ROOT_PATH}/six_dof'
if ROOT_PATH not in sys.path: sys.path.insert(0, ROOT_PATH)
if SIX_DOF_PATH not in sys.path: sys.path.insert(0, SIX_DOF_PATH)
import ompl_kinodynamic_rrt_6dof as rrt6d
stats = rrt6d.plan_6dof_kinodynamic_rrt((-8.0,-8.0,-3.0),(0.0,0.0,0.0),(8.0,7.0,2.0),(0.0,0.0,0.0), planning_time=20.0)
print(f"\nFinal: Solved={stats['solved']}, Time={stats['planning_time']:.2f}s, Path={stats['linear_length']:.1f}m")

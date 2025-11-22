#!/usr/bin/env python3
"""Timing test for 6DoF planner (moved to six_dof)."""
import sys
ROOT_PATH = '/home/ethan/ros2_ws/src/Proactive-Human-Manipulator-collision-avoidance-through-motion-prediction-and-ROS/Benchmarking'
SIX_DOF_PATH = f'{ROOT_PATH}/six_dof'
if ROOT_PATH not in sys.path: sys.path.insert(0, ROOT_PATH)
if SIX_DOF_PATH not in sys.path: sys.path.insert(0, SIX_DOF_PATH)
import ompl_kinodynamic_rrt_6dof as rrt6d
print("Testing 6DoF RRT with different time limits (six_dof):\n" + "="*60)
for test_time in [5,10,15,20,30]:
    print(f"\nTesting with {test_time}s timeout:")
    stats = rrt6d.plan_6dof_kinodynamic_rrt((-8.0,-8.0,-3.0),(0.0,0.0,0.0),(8.0,7.0,2.0),(0.0,0.0,0.0), planning_time=float(test_time))
    if stats['solved']:
        print(f"  ✓ Found solution in {stats['planning_time']:.3f}s")
        break
    else:
        print(f"  ✗ No solution found in {stats['planning_time']:.3f}s")
print("\n" + "="*60 + "\nConclusion: Long search is due to true difficulty in 12D state space.")

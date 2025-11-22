#!/usr/bin/env python3
"""
Quick direct comparison without subprocess overhead (post-reorg)
"""

import sys
import time
import numpy as np

ROOT_PATH = '/home/ethan/ros2_ws/src/Proactive-Human-Manipulator-collision-avoidance-through-motion-prediction-and-ROS/Benchmarking'
STEERING_PATH = f'{ROOT_PATH}/steering_components'
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
if STEERING_PATH not in sys.path:
    sys.path.insert(0, STEERING_PATH)

print("="*80)
print(" " * 25 + "RRT METHOD COMPARISON")
print("="*80)
print("\nRunning 5 trials of each method...\n")

start_pos = (-8.0, -8.0)
goal_pos = (8.0, 7.0)
start_vel = (0.0, 0.0)
goal_vel = (0.0, 0.0)

print(f"Problem: Start {start_pos} → Goal {goal_pos}")
print("-"*80)

print("\n🔵 CONTROL SAMPLING RRT (OMPL RRT*):")
print("-"*80)
import ompl_kinodynamic_rrt_star_2d as control_rrt

control_results = []
for i in range(5):
    print(f"\nTrial {i+1}/5...", end=" ")
    solution_path, solution_controls, stats = control_rrt.plan_kinodynamic_rrt_star(
        start_pos=start_pos, goal_pos=goal_pos,
        start_vel=start_vel, goal_vel=goal_vel,
        planning_time=10.0
    )
    if stats['solved']:
        control_results.append(stats)
        print(f"✓ Time: {stats['planning_time']:.3f}s, Path: {stats['path_length']:.2f}m, States: {stats['num_states']}")
    else:
        print("✗ Failed")

if control_results:
    avg_time = np.mean([r['planning_time'] for r in control_results])
    avg_path = np.mean([r['path_length'] for r in control_results])
    avg_states = np.mean([r['num_states'] for r in control_results])
    success_rate = len(control_results) / 5 * 100
    print(f"\n📊 STATISTICS (n={len(control_results)}):")
    print(f"   Success Rate:  {success_rate:.0f}%")
    print(f"   Avg Time:      {avg_time:.3f}s (±{np.std([r['planning_time'] for r in control_results]):.3f}s)")
    print(f"   Avg Path:      {avg_path:.2f}m (±{np.std([r['path_length'] for r in control_results]):.2f}m)")
    print(f"   Avg States:    {avg_states:.1f} (±{np.std([r['num_states'] for r in control_results]):.1f})")

print("\n" + "="*80)
print("\n🔴 STEERING-BASED RRT (Analytical TPBVP):")
print("-"*80)
import ompl_kinodynamic_rrt_steering as steering_rrt

steering_results = []
for i in range(5):
    print(f"\nTrial {i+1}/5...", end=" ")
    solution_path, solution_controls, stats = steering_rrt.plan_with_steering(
        start_pos=start_pos, goal_pos=goal_pos,
        start_vel=start_vel, goal_vel=goal_vel,
        planning_time=10.0
    )
    if stats['solved']:
        steering_results.append(stats)
        print(f"✓ Time: {stats['planning_time']:.3f}s, Path: {stats['path_length']:.2f}m, States: {stats['num_states']}")
    else:
        print("✗ Failed")

if steering_results:
    avg_time = np.mean([r['planning_time'] for r in steering_results])
    avg_path = np.mean([r['path_length'] for r in steering_results])
    avg_states = np.mean([r['num_states'] for r in steering_results])
    success_rate = len(steering_results) / 5 * 100
    print(f"\n📊 STATISTICS (n={len(steering_results)}):")
    print(f"   Success Rate:  {success_rate:.0f}%")
    print(f"   Avg Time:      {avg_time:.3f}s (±{np.std([r['planning_time'] for r in steering_results]):.3f}s)")
    print(f"   Avg Path:      {avg_path:.2f}m (±{np.std([r['path_length'] for r in steering_results]):.2f}m)")
    print(f"   Avg States:    {avg_states:.1f} (±{np.std([r['num_states'] for r in steering_results]):.1f})")

print("\n" + "="*80)
print(" " * 30 + "COMPARISON")
print("="*80)

if control_results and steering_results:
    control_avg_time = np.mean([r['planning_time'] for r in control_results])
    steering_avg_time = np.mean([r['planning_time'] for r in steering_results])
    control_avg_path = np.mean([r['path_length'] for r in control_results])
    steering_avg_path = np.mean([r['path_length'] for r in steering_results])
    control_avg_states = np.mean([r['num_states'] for r in control_results])
    steering_avg_states = np.mean([r['num_states'] for r in steering_results])

    print(f"\n{'Metric':<30} {'Control RRT':<20} {'Steering RRT':<20} {'Speedup':<10}")
    print("-"*80)
    print(f"{'Planning Time':<30} {control_avg_time:.3f}s{'':<15} {steering_avg_time:.3f}s{'':<15} {steering_avg_time/control_avg_time:.1f}x slower")
    print(f"{'Path Length':<30} {control_avg_path:.2f}m{'':<15} {steering_avg_path:.2f}m{'':<15} {abs(1-steering_avg_path/control_avg_path)*100:.1f}% diff")
    print(f"{'Number of States':<30} {control_avg_states:.0f}{'':<19} {steering_avg_states:.0f}{'':<19} {steering_avg_states/control_avg_states:.1f}x more")

    print("\n" + "="*80)
    print("CONCLUSIONS:")
    print("-"*80)
    print(f"• Control Sampling RRT is {steering_avg_time/control_avg_time:.0f}x FASTER")
    print(f"• Both methods find similar path lengths (~{control_avg_path:.0f}m)")
    print(f"• Control RRT uses FEWER states ({control_avg_states:.0f} vs {steering_avg_states:.0f})")
    print(f"\nREASONS:")
    print("  → Control Sampling uses optimized C++ OMPL implementation")
    print("  → Steering-based is pure Python with analytical TPBVP solving")
    print("  → Control sampling explores space more efficiently")
    print("  → Steering requires more iterations due to exact trajectory computation")
    print("="*80)

print("\n✓ Comparison complete!\n")

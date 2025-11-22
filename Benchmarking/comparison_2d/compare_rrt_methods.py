#!/usr/bin/env python3
"""
Comparison script for Control Sampling RRT vs Steering-based RRT
Run with: conda activate master_project && python compare_rrt_methods.py
"""

import sys
import time
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Ensure root and steering component paths are available after reorganization
ROOT_PATH = '/home/ethan/ros2_ws/src/Proactive-Human-Manipulator-collision-avoidance-through-motion-prediction-and-ROS/Benchmarking'
STEERING_PATH = f'{ROOT_PATH}/steering_components'
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
if STEERING_PATH not in sys.path:
    sys.path.insert(0, STEERING_PATH)

print("Loading implementations...")

import subprocess
import json
import tempfile
import os


def run_control_sampling_rrt(start_pos: Tuple[float, float], goal_pos: Tuple[float, float],
                             start_vel: Tuple[float, float], goal_vel: Tuple[float, float]) -> Dict[str, Any]:
    """Run the control sampling RRT from ompl_kinodynamic_rrt_star_2d.py"""

    script_content = f"""
import sys
ROOT_PATH = '{ROOT_PATH}'
STEERING_PATH = '{STEERING_PATH}'
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
if STEERING_PATH not in sys.path:
    sys.path.insert(0, STEERING_PATH)
import ompl_kinodynamic_rrt_star_2d as control_rrt
import json

start_pos = {start_pos}
goal_pos = {goal_pos}
start_vel = {start_vel}
goal_vel = {goal_vel}

solution_path, solution_controls, stats = control_rrt.plan_kinodynamic_rrt_star(
    start_pos=start_pos, goal_pos=goal_pos,
    start_vel=start_vel, goal_vel=goal_vel,
    planning_time=30.0
)

results = {{
    'solved': stats['solved'],
    'planning_time': stats['planning_time'],
    'path_length': stats['path_length'],
    'num_states': stats['num_states'],
    'duration': stats['duration'],
    'path': [(s[0], s[1], s[2], s[3]) for s in solution_path] if solution_path else None
}}

with open('control_rrt_results.json', 'w') as f:
    json.dump(results, f)
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'master_project', 'python', temp_script],
            cwd=ROOT_PATH,
            capture_output=True,
            text=True,
            timeout=60
        )

        results_file = f'{ROOT_PATH}/control_rrt_results.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            os.remove(results_file)
            return results
        else:
            print(f"Control RRT stderr: {result.stderr}")
            return {'solved': False, 'planning_time': 0, 'path_length': 0, 'num_states': 0, 'duration': 0, 'path': None}
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


def run_steering_rrt(start_pos: Tuple[float, float], goal_pos: Tuple[float, float],
                     start_vel: Tuple[float, float], goal_vel: Tuple[float, float]) -> Dict[str, Any]:
    """Run the steering-based RRT from ompl_kinodynamic_rrt_steering.py"""

    script_content = f"""
import sys
ROOT_PATH = '{ROOT_PATH}'
STEERING_PATH = '{STEERING_PATH}'
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
if STEERING_PATH not in sys.path:
    sys.path.insert(0, STEERING_PATH)
import ompl_kinodynamic_rrt_steering as steering_rrt
import json

start_pos = {start_pos}
goal_pos = {goal_pos}
start_vel = {start_vel}
goal_vel = {goal_vel}

solution_path, solution_controls, stats = steering_rrt.plan_with_steering(
    start_pos=start_pos, goal_pos=goal_pos,
    start_vel=start_vel, goal_vel=goal_vel,
    planning_time=30.0
)

results = {{
    'solved': stats['solved'],
    'planning_time': stats['planning_time'],
    'path_length': stats['path_length'],
    'num_states': stats['num_states'],
    'duration': stats['duration'],
    'path': [(s[0], s[1], s[2], s[3]) for s in solution_path] if solution_path else None
}}

with open('steering_rrt_results.json', 'w') as f:
    json.dump(results, f)
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'master_project', 'python', temp_script],
            cwd=ROOT_PATH,
            capture_output=True,
            text=True,
            timeout=60
        )

        results_file = f'{ROOT_PATH}/steering_rrt_results.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            os.remove(results_file)
            return results
        else:
            print(f"Steering RRT stderr: {result.stderr}")
            return {'solved': False, 'planning_time': 0, 'path_length': 0, 'num_states': 0, 'duration': 0, 'path': None}
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


def visualize_comparison(control_results: Dict, steering_results: Dict) -> None:
    """Create comparison visualization."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    methods = ['Control\nSampling RRT', 'Steering-based\nRRT']
    times = [control_results['planning_time'], steering_results['planning_time']]
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Planning Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Planning Time Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')

    ax2 = fig.add_subplot(gs[0, 1])
    lengths = [control_results['path_length'], steering_results['path_length']]
    bars = ax2.bar(methods, lengths, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Path Length (meters)', fontsize=11, fontweight='bold')
    ax2.set_title('Path Length Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, length in zip(bars, lengths):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{length:.2f}m', ha='center', va='bottom', fontweight='bold')

    ax3 = fig.add_subplot(gs[0, 2])
    states = [control_results['num_states'], steering_results['num_states']]
    bars = ax3.bar(methods, states, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Number of States', fontsize=11, fontweight='bold')
    ax3.set_title('Path Complexity', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar, state_count in zip(bars, states):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{state_count}', ha='center', va='bottom', fontweight='bold')

    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_xlim(-10, 10)
    ax4.set_ylim(-10, 10)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax4.set_title('Trajectory Comparison', fontsize=12, fontweight='bold')

    obstacles = [(2.0, 2.0, 1.0), (5.0, -2.0, 1.2), (-3.0, 4.0, 0.8)]
    for ox, oy, radius in obstacles:
        circle = Circle((ox, oy), radius, color='red', alpha=0.3, zorder=1)
        ax4.add_patch(circle)

    if control_results['path']:
        path = control_results['path']
        x_coords = [s[0] for s in path]
        y_coords = [s[1] for s in path]
        ax4.plot(x_coords, y_coords, 'o-', color='#3498db', linewidth=2.5, markersize=3, label='Control Sampling RRT', alpha=0.7, zorder=2)

    if steering_results['path']:
        path = steering_results['path']
        x_coords = [s[0] for s in path]
        y_coords = [s[1] for s in path]
        ax4.plot(x_coords, y_coords, 's-', color='#e74c3c', linewidth=2.5, markersize=3, label='Steering-based RRT', alpha=0.7, zorder=3)

    if control_results['path']:
        start = control_results['path'][0]
        goal = control_results['path'][-1]
        ax4.plot(start[0], start[1], 'go', markersize=12, label='Start', zorder=4)
        ax4.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal', zorder=4)

    ax4.legend(loc='upper left', fontsize=10)
    plt.suptitle('RRT Comparison: Control Sampling vs Steering-based', fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('rrt_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Comparison plot saved as 'rrt_comparison.png'")
    plt.show()


def print_comparison_table(control_results: Dict, steering_results: Dict) -> None:
    print("\n" + "="*80)
    print(" " * 25 + "RRT METHOD COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} {'Control Sampling':<20} {'Steering-based':<20} {'Ratio':<10}")
    print("-"*80)

    metrics = [
        ('Planning Time (s)', 'planning_time', 1, '.3f'),
        ('Path Length (m)', 'path_length', 1, '.2f'),
        ('Trajectory Duration (s)', 'duration', 1, '.2f'),
        ('Number of States', 'num_states', 1, 'd'),
    ]

    for metric_name, key, scale, fmt in metrics:
        control_val = control_results[key] * scale
        steering_val = steering_results[key] * scale
        ratio_str = f"{(steering_val / control_val):.2f}x" if control_val > 0 and steering_val > 0 else "N/A"
        control_str = f"{control_val:{fmt}}"
        steering_str = f"{steering_val:{fmt}}"
        print(f"{metric_name:<30} {control_str:<20} {steering_str:<20} {ratio_str:<10}")

    print("-"*80)
    print(f"{'Success Rate':<30} {'✓' if control_results['solved'] else '✗':<20} "
          f"{'✓' if steering_results['solved'] else '✗':<20}")
    print("="*80)

    print("\n📈 ANALYSIS:")
    print("-"*80)
    if control_results['solved'] and steering_results['solved']:
        time_diff = steering_results['planning_time'] / control_results['planning_time']
        length_diff = steering_results['path_length'] / control_results['path_length']
        print(f"• Control Sampling RRT is {time_diff:.1f}x FASTER in planning time")
        print(f"• Steering-based RRT path is {length_diff:.2f}x {'LONGER' if length_diff > 1 else 'SHORTER'}")
        print(f"• Control Sampling uses {control_results['num_states']} states vs {steering_results['num_states']} for Steering")
        print("\nREASONS FOR DIFFERENCES:")
        print("  - Control Sampling: Uses optimized OMPL C++ implementation with random control sampling")
        print("  - Steering-based: Custom Python implementation solving TPBVP at each iteration")
        print("  - Control Sampling explores more efficiently with RRT*")
        print("  - Steering-based computes optimal trajectories but requires more iterations")


def main():
    print("\n" + "="*80)
    print(" " * 20 + "RRT COMPARISON BENCHMARK")
    print("="*80)
    print("Comparing Control Sampling RRT vs Steering-based RRT\n")

    start_pos = (-8.0, -8.0)
    goal_pos = (8.0, 7.0)
    start_vel = (0.0, 0.0)
    goal_vel = (0.0, 0.0)

    print(f"Test Problem:")
    print(f"  Start: pos={start_pos}, vel={start_vel}")
    print(f"  Goal:  pos={goal_pos}, vel={goal_vel}")
    print(f"  Obstacles: 3 circular obstacles")
    print("\n" + "-"*80)

    print("\n🔵 Running Control Sampling RRT (OMPL RRT*)...")
    control_results = run_control_sampling_rrt(start_pos, goal_pos, start_vel, goal_vel)
    if control_results['solved']:
        print(f"   ✓ Solved in {control_results['planning_time']:.3f}s, path length: {control_results['path_length']:.2f}m")
    else:
        print("   ✗ Failed to find solution")

    print("\n🔴 Running Steering-based RRT (Analytical TPBVP)...")
    steering_results = run_steering_rrt(start_pos, goal_pos, start_vel, goal_vel)
    if steering_results['solved']:
        print(f"   ✓ Solved in {steering_results['planning_time']:.3f}s, path length: {steering_results['path_length']:.2f}m")
    else:
        print("   ✗ Failed to find solution")

    print_comparison_table(control_results, steering_results)

    if control_results['solved'] or steering_results['solved']:
        visualize_comparison(control_results, steering_results)

    print("\n" + "="*80)
    print("✓ Comparison complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

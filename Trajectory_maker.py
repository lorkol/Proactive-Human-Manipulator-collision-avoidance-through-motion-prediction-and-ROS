import json
import numpy as np

# --- 1. CONFIGURATION ---
JOINT_NAMES = ['CLAV', 'C7', 'RSHO', 'LSHO', 'LAEL', 'RAEL', 'LWPS', 'RWPS',
               'L3', 'LHIP', 'RHIP', 'LKNE', 'RKNE', 'LHEE', 'RHEE']
               
# BASE_Y_OFFSET_MM: Sets the human's average distance from the robot base (Y=0).
# A tighter clearance of -300mm to -500mm ensures the robot's arm moving from 
# Q1=(0.0, -1.0, 1.0, ...) to Q2=(-2.0, -1.0, 1.0, ...) will pass close by.
BASE_Y_OFFSET_MM = -250

# Approximate standing pose (mm)
# Assuming Z is UP, X is lateral (left/right), Y is forward/depth (robot is at (0,0,0))
# The INITIAL_POSE now defines a person standing at the tight BASE_Y_OFFSET_MM distance.
INITIAL_POSE_STAND = np.array([
    [0, BASE_Y_OFFSET_MM, 1550],  # CLAV (0)
    [0, BASE_Y_OFFSET_MM, 1600],  # C7 (1)
    [200, BASE_Y_OFFSET_MM, 1400],  # RSHO (2)
    [-200, BASE_Y_OFFSET_MM, 1400],  # LSHO (3)
    [250, BASE_Y_OFFSET_MM, 1100],  # LAEL (4)
    [-250, BASE_Y_OFFSET_MM, 1100], # RAEL (5)
    [250, BASE_Y_OFFSET_MM, 800],  # LWPS (6)
    [-250, BASE_Y_OFFSET_MM, 800], # RWPS (7)
    [0, BASE_Y_OFFSET_MM, 1200],  # L3 (8) - interpolated spine
    [100, BASE_Y_OFFSET_MM, 950],  # LHIP (9)
    [-100, BASE_Y_OFFSET_MM, 950], # RHIP (10)
    [100, BASE_Y_OFFSET_MM, 450],  # LKNE (11)
    [-100, BASE_Y_OFFSET_MM, 450], # RKNE (12)
    [100, BASE_Y_OFFSET_MM, 0],    # LHEE (13)
    [-100, BASE_Y_OFFSET_MM, 0]    # RHEE (14)
], dtype=np.float64) # **CRITICAL FIX: Ensure float type to avoid casting error**

# --- 2. CORE UTILITY FUNCTIONS ---
def generate_trajectory(start_pose, frames, duration, motion_func):
    """Generates a trajectory given a motion function."""
    trajectory = []
    time_steps = np.linspace(0, duration, frames)
    for t in time_steps:
        new_pose = motion_func(start_pose, t, duration)
        # Convert pose back to the list format for JSON export
        trajectory.append({
            "time_s": float(t),
            "joint_poses_mm": new_pose.round(3).tolist()
        })
    return trajectory

# --- 3. SCENARIO MOTION FUNCTIONS (1-3) ---

# S1: Simple Passing (Translate the entire body)
def motion_pass_by(pose, t, duration, start_x=-1500, end_x=1500, y_offset=BASE_Y_OFFSET_MM):
    progress = t / duration
    current_x = start_x + (end_x - start_x) * progress
    
    # We use a base pose near the robot, then shift it sideways (X-axis)
    translated_pose = pose.copy()
    translated_pose[:, 0] += current_x # Shift X
    translated_pose[:, 1] = y_offset # Set Y (depth) to ensure collision risk
    
    # Simple leg animation
    if (progress * 10) % 2 < 1:
        translated_pose[[11, 13], 2] += 20 * np.sin(progress * np.pi * 5)
    return translated_pose

# S2: Reach/Intrusion (Translate a single hand relative to the shoulder/C7)
def motion_reach(pose, t, duration, joint_idx, intrude_y_mm=200):
    if t < duration * 0.2: # Wait period
        return pose # Static pose close to the robot
        
    progress = (t - duration * 0.2) / (duration * 0.8)
    intruding_pose = pose.copy()
    
    # Y-axis is the intrusion direction (towards the robot, Y=0)
    current_y_delta = intrude_y_mm * progress 
    
    # Intrusion is only applied to the specified joint (e.g., WRIST)
    # The wrist's initial Y position (pose[joint_idx, 1]) is already BASE_Y_OFFSET_MM
    # We move the wrist from BASE_Y_OFFSET_MM towards BASE_Y_OFFSET_MM + current_y_delta
    # Since BASE_Y_OFFSET_MM is negative, moving towards zero is positive delta
    intruding_pose[joint_idx, 1] += current_y_delta
    
    # Keep the rest of the arm attached by moving Elbow/Shoulder slightly
    if joint_idx in [6, 7]: # LWPS or RWPS
        shoulder_idx = joint_idx - 4
        intruding_pose[shoulder_idx, 1] += current_y_delta * 0.3 # Shoulder moves 30%
        
    return intruding_pose

# S3: Prolonged Interaction (Shift torso/hips)
def motion_lean_in(pose, t, duration, max_lean_mm=BASE_Y_OFFSET_MM + 200):
    progress = t / duration
    # This creates a slow, sustained lean toward the robot's working volume (Y=0)
    current_lean_y = max_lean_mm * np.sin(progress * np.pi) 
    
    lean_pose = pose.copy() 
    
    # Apply lean (Y-axis translation) to all upper body joints (above hips, 0 to 10)
    lean_pose[:11, 1] += current_lean_y
    
    return lean_pose

# S4: Slow Close Interaction (Moving between predefined positions with smooth transitions)
def motion_slow_close_interaction(pose, t, duration):
    """
    Creates a slow, deliberate movement pattern between predefined positions close to the robot.
    Uses smooth sinusoidal transitions for natural movement.
    """
    # Define key positions relative to the base pose
    positions = {
        'center': np.array([0, 0, 0]),
        'left': np.array([-300, 50, 0]),    # Left side, slightly closer
        'right': np.array([300, 50, 0]),    # Right side, slightly closer
        'forward': np.array([0, 100, 0]),   # Forward position, closest to robot
    }
    
    # Divide the duration into segments
    segment_duration = duration / 4
    current_segment = int((t / duration) * 4)
    segment_progress = (t % segment_duration) / segment_duration
    
    # Smooth transition using sine wave (easier acceleration/deceleration)
    smooth_progress = 0.5 * (1 - np.cos(segment_progress * np.pi))
    
    # Define the movement sequence
    sequence = ['center', 'left', 'forward', 'right']
    current_pos = sequence[current_segment % len(sequence)]
    next_pos = sequence[(current_segment + 1) % len(sequence)]
    
    # Interpolate between positions
    delta = positions[current_pos] * (1 - smooth_progress) + positions[next_pos] * smooth_progress
    
    # Apply movement to the pose
    modified_pose = pose.copy()
    
    # Apply position changes to upper body (joints 0-8)
    modified_pose[:9, 0] += delta[0]  # X translation
    modified_pose[:9, 1] = BASE_Y_OFFSET_MM + delta[1]  # Y position (closer/further)
    modified_pose[:9, 2] += delta[2]  # Z translation
    
    # Add subtle swaying motion to make it more natural
    sway = 20 * np.sin(t * 2 * np.pi / duration)  # Gentle 20mm sway
    modified_pose[:9, 0] += sway
    
    # Add subtle breathing motion
    breathing = 10 * np.sin(t * 6 * np.pi / duration)  # Subtle up/down motion
    modified_pose[:9, 2] += breathing
    
    return modified_pose

# Helper function to combine multiple motion patterns
def motion_composite(pose, t, duration, motions):
    """
    Combines multiple motion patterns with weights.
    motions: list of tuples (motion_func, weight)
    """
    result_pose = pose.copy()
    for motion_func, weight in motions:
        result_pose = result_pose * (1 - weight) + motion_func(pose, t, duration) * weight
    return result_pose

# --- 4. GENERATE TRAJECTORIES ---
all_trajectories = []
total_count = 0
# Use the adjusted stand pose as the base for all movements
INITIAL_POSE = INITIAL_POSE_STAND 

# S4: Slow Close Interactions (10 trajectories with different variations)
for i in range(5):  # 5 basic slow close movements
    total_count += 1
    duration = 10.0  # 10 seconds for slower movement
    frames = 200    # More frames for smoother motion
    
    all_trajectories.append({
        "scenario_id": f"S4-{total_count}-SlowClose",
        "description": "Slow movement pattern with closer interaction",
        "trajectory": generate_trajectory(INITIAL_POSE, frames, duration, motion_slow_close_interaction)
    })

# Add some composite movements combining slow close with reaching
for i in range(5):  # 5 composite movements
    total_count += 1
    duration = 12.0  # 12 seconds for even slower movement
    frames = 240    # More frames for smoother motion
    
    # Create composite motion with different reach patterns
    reach_joint = 6 + i % 2  # Alternates between left (6) and right (7) hand
    def composite_motion(pose, t, d, reach_joint=reach_joint):
        return motion_composite(pose, t, d, [
            (motion_slow_close_interaction, 0.7),
            (lambda p, t, d: motion_reach(p, t, d, joint_idx=reach_joint, intrude_y_mm=100), 0.3)
        ])
    
    all_trajectories.append({
        "scenario_id": f"S4-{total_count}-CompositeReach",
        "description": f"Slow close movement with {'left' if reach_joint==6 else 'right'} hand reaching",
        "trajectory": generate_trajectory(INITIAL_POSE, frames, duration, composite_motion)
    })

# S1: Simple Passing (20 trajectories) - These are mostly for testing collision *detection*
for i in range(10): # 10 Fast (3s)
    total_count += 1
    all_trajectories.append({
        "scenario_id": f"S1-{total_count}-Fast",
        "description": f"Fast walk-by (3s) at close clearance (Y={BASE_Y_OFFSET_MM}mm).",
        "trajectory": generate_trajectory(INITIAL_POSE, 30, 3.0, motion_pass_by)
    })
for i in range(10): # 10 Slow (6s)
    total_count += 1
    all_trajectories.append({
        "scenario_id": f"S1-{total_count}-Slow",
        "description": f"Slow walk-by (6s) at very tight clearance (Y={BASE_Y_OFFSET_MM + 150}mm).",
        "trajectory": generate_trajectory(INITIAL_POSE, 60, 6.0, 
                                               lambda p, t, d: motion_pass_by(p, t, d, y_offset=BASE_Y_OFFSET_MM + 150))
    })

# S2: Reach/Intrusion (15 trajectories) - Focus on the arm
for i in range(5): # Right Wrist Intrusion
    total_count += 1
    all_trajectories.append({
        "scenario_id": f"S2-{total_count}-R_Reach",
        "description": "Right wrist intrusion (RWPS, index 7) reaching 200mm towards the robot.",
        "trajectory": generate_trajectory(INITIAL_POSE, 40, 4.0, 
                                               lambda p, t, d: motion_reach(p, t, d, joint_idx=7, intrude_y_mm=200))
    })
for i in range(5): # Left Wrist Intrusion
    total_count += 1
    all_trajectories.append({
        "scenario_id": f"S2-{total_count}-L_Reach",
        "description": "Left wrist intrusion (LWPS, index 6) reaching 300mm towards the robot.",
        "trajectory": generate_trajectory(INITIAL_POSE, 40, 4.0, 
                                               lambda p, t, d: motion_reach(p, t, d, joint_idx=6, intrude_y_mm=300))
    })
for i in range(5): # Two-Hand Reach
    total_count += 1
    def two_hand_motion(pose, t, d):
        p1 = motion_reach(pose, t, d, joint_idx=6, intrude_y_mm=200)
        return motion_reach(p1, t, d, joint_idx=7, intrude_y_mm=200)
        
    all_trajectories.append({
        "scenario_id": f"S2-{total_count}-Two_Reach",
        "description": "Two-hand symmetric reach 200mm towards the robot.",
        "trajectory": generate_trajectory(INITIAL_POSE, 50, 5.0, two_hand_motion)
    })

# S3: Prolonged Interaction (15 trajectories) - Torso/Capsule test
for i in range(15): # Lean-in
    total_count += 1
    # Randomize the maximum lean amount
    max_lean = np.random.randint(100, 200)
    all_trajectories.append({
        "scenario_id": f"S3-{total_count}-Lean_In",
        "description": f"Slow, prolonged lean-in/sway (5s) with max {max_lean}mm lean towards robot.",
        "trajectory": generate_trajectory(INITIAL_POSE, 50, 5.0, 
                                               lambda p, t, d: motion_lean_in(p, t, d, max_lean_mm=max_lean))
    })

# --- 5. EXPORT TO JSON FILE ---
filename = "human_trajectories_50.json"
with open(filename, 'w') as f:
    json.dump(all_trajectories, f, indent=2)

print(f"\nSuccessfully generated {total_count} trajectories with human close to robot and saved to {filename}")
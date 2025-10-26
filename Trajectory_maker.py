import json
import numpy as np

# --- 1. CONFIGURATION ---
JOINT_NAMES = ['CLAV', 'C7', 'RSHO', 'LSHO', 'LAEL', 'RAEL', 'LWPS', 'RWPS',
               'L3', 'LHIP', 'RHIP', 'LKNE', 'RKNE', 'LHEE', 'RHEE']
# Approximate standing pose (mm)
# Assuming Z is UP, X is lateral (left/right), Y is forward/depth (robot is at (0,0,0))
# Human starts at Y=-1000 (1 meter in front of the robot base)
INITIAL_POSE = np.array([
    [0, 0, 1550],  # CLAV (0)
    [0, 0, 1600],  # C7 (1)
    [200, 0, 1400],  # RSHO (2)
    [-200, 0, 1400],  # LSHO (3)
    [250, 0, 1100],  # LAEL (4)
    [-250, 0, 1100],  # RAEL (5)
    [250, 0, 800],  # LWPS (6)
    [-250, 0, 800],  # RWPS (7)
    [0, 0, 1200],  # L3 (8) - interpolated spine
    [100, 0, 950],  # LHIP (9)
    [-100, 0, 950],  # RHIP (10)
    [100, 0, 450],  # LKNE (11)
    [-100, 0, 450],  # RKNE (12)
    [100, 0, 0],  # LHEE (13)
    [-100, 0, 0]  # RHEE (14)
])


# --- 2. CORE UTILITY FUNCTIONS ---
def generate_trajectory(start_pose, frames, duration, motion_func):
    """Generates a trajectory given a motion function."""
    trajectory = []
    time_steps = np.linspace(0, duration, frames)
    for t in time_steps:
        new_pose = motion_func(start_pose, t, duration)
        trajectory.append({
            "time_s": float(t),
            "joint_poses_mm": new_pose.round(3).tolist()
        })
    return trajectory


# --- 3. SCENARIO MOTION FUNCTIONS (1-3) ---

# S1: Simple Passing (Translate the entire body)
def motion_pass_by(pose, t, duration, start_x=-1500, end_x=1500, y_offset=-1000):
    progress = t / duration
    current_x = start_x + (end_x - start_x) * progress
    # Base pose translation
    translated_pose = pose + np.array([current_x, y_offset, 0])
    # Slight dynamic step (simulated by lifting one knee/heel)
    if (progress * 10) % 2 < 1:
        translated_pose[[11, 13], 2] += 20 * np.sin(progress * np.pi * 5)
    return translated_pose


# S2: Reach/Intrusion (Translate a single hand relative to the shoulder/C7)
def motion_reach(pose, t, duration, joint_idx, start_z_mm=800, end_z_mm=400, y_intrude=200):
    if t < duration * 0.2:  # Wait period
        return pose + np.array([0, -1000, 0])
    progress = (t - duration * 0.2) / (duration * 0.8)

    # Intrusion is only applied to the specified joint (e.g., WRIST)
    intruding_pose = pose.copy()

    # Interpolate Z and Y position (forward and down to reach)
    current_z = start_z_mm + (end_z_mm - start_z_mm) * progress
    current_y = 0 + y_intrude * progress

    # Calculate shoulder position (e.g., RSHO) for relative movement
    shoulder_idx = joint_idx - 5 if joint_idx == 7 else joint_idx - 4

    # Set the new position for the wrist
    target_pos = intruding_pose[shoulder_idx].copy()
    target_pos[2] = current_z
    target_pos[1] += current_y  # Intrusion in Y (depth)

    intruding_pose[joint_idx] = target_pos

    # Base pose translation (human is standing)
    return intruding_pose + np.array([0, -1000, 0])


# S3: Prolonged Interaction (Shift torso/hips)
def motion_lean_in(pose, t, duration, max_lean_mm=150):
    progress = t / duration
    current_lean_y = max_lean_mm * np.sin(progress * np.pi)

    # Cast to float64 before performing addition with the float value
    # Apply lean (Y-axis translation) to all upper body joints (above waist)
    lean_pose = pose.astype(np.float64).copy() + np.array([0, -1000, 0], dtype=np.float64)
    # lean_pose = pose.copy() + np.array([0, -1000, 0])  # Base pos

    # Apply lean only to the upper body joints (0 to 8)
    lean_pose[:9, 1] += current_lean_y

    # Shift hips slightly as well
    lean_pose[[9, 10], 1] += current_lean_y * 0.5

    return lean_pose


# --- 4. GENERATE 50 TRAJECTORIES ---
all_trajectories = []
total_count = 0

# S1: Simple Passing (20 trajectories)
for i in range(10):  # 10 Fast (3s)
    total_count += 1
    all_trajectories.append({
        "scenario_id": f"S1-{total_count}-Fast",
        "description": f"Fast walk-by (3s) at medium clearance (Y={-1000}mm).",
        "trajectory_data": generate_trajectory(INITIAL_POSE, 30, 3.0, motion_pass_by)
    })
for i in range(10):  # 10 Slow (6s)
    total_count += 1
    all_trajectories.append({
        "scenario_id": f"S1-{total_count}-Slow",
        "description": f"Slow walk-by (6s) at close clearance (Y={-800}mm).",
        "trajectory_data": generate_trajectory(INITIAL_POSE, 60, 6.0,
                                               lambda p, t, d: motion_pass_by(p, t, d, y_offset=-800))
    })

# S2: Reach/Intrusion (15 trajectories)
for i in range(5):  # Right Wrist Intrusion
    total_count += 1
    all_trajectories.append({
        "scenario_id": f"S2-{total_count}-R_Reach",
        "description": "Right wrist intrusion (RWPS, index 7) reaching forward 200mm (Y intrusion).",
        "trajectory_data": generate_trajectory(INITIAL_POSE, 40, 4.0,
                                               lambda p, t, d: motion_reach(p, t, d, joint_idx=7, y_intrude=200))
    })
for i in range(5):  # Left Wrist Intrusion
    total_count += 1
    all_trajectories.append({
        "scenario_id": f"S2-{total_count}-L_Reach",
        "description": "Left wrist intrusion (LWPS, index 6) reaching forward 300mm (Y intrusion).",
        "trajectory_data": generate_trajectory(INITIAL_POSE, 40, 4.0,
                                               lambda p, t, d: motion_reach(p, t, d, joint_idx=6, y_intrude=300))
    })
for i in range(5):  # Two-Hand Reach (use a combined motion)
    total_count += 1


    def two_hand_motion(pose, t, d):
        p1 = motion_reach(pose, t, d, joint_idx=6, y_intrude=200)
        return motion_reach(p1, t, d, joint_idx=7, y_intrude=200)


    all_trajectories.append({
        "scenario_id": f"S2-{total_count}-Two_Reach",
        "description": "Two-hand symmetric reach forward (200mm Y intrusion).",
        "trajectory_data": generate_trajectory(INITIAL_POSE, 50, 5.0, two_hand_motion)
    })

# S3: Prolonged Interaction (15 trajectories)
for i in range(15):  # Lean-in
    total_count += 1
    all_trajectories.append({
        "scenario_id": f"S3-{total_count}-Lean_In",
        "description": f"Slow, prolonged lean-in/sway (5s) for sustained APF stress (max {np.random.randint(100, 200)}mm lean).",
        "trajectory_data": generate_trajectory(INITIAL_POSE, 50, 5.0,
                                               lambda p, t, d: motion_lean_in(p, t, d,
                                                                              max_lean_mm=np.random.randint(100, 200)))
    })

# --- 5. EXPORT TO JSON FILE ---
filename = "human_trajectories_50.json"
with open(filename, 'w') as f:
    json.dump(all_trajectories, f, indent=2)

print(f"\nSuccessfully generated {total_count} trajectories and saved to {filename}")
import time
import cv2
import numpy as np
import mediapipe as mp
from pyorbbecsdk import Config, OBSensorType, OBFormat, Pipeline, OBAlignMode
from utils import frame_to_bgr_image
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


colorx= 1280
colory= 960

ESC_KEY = 27
MIN_DEPTH = 20
MAX_DEPTH = 10000
DISPLAY_SCALE = 0.5
stop_accel=False
JOINT_NAMES = ['CLAV', 'C7', 'RSHO', 'LSHO', 'LAEL', 'RAEL', 'LWPS', 'RWPS',
               'L3', 'LHIP', 'RHIP', 'LKNE', 'RKNE', 'LHEE', 'RHEE']

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


global Ax,Ay,Az

def on_accel_frame_callback(frame):
    global Ax,Ay, Az
    accel_frame: AccelFrame = frame.as_accel_frame()
    if accel_frame is not None:
        Ax,Ay,Az=accel_frame.get_x(), accel_frame.get_y(),accel_frame.get_z()
        Ax,Ay,Az=-Ay,Az,-Ax #imu frame to cam frame
        

def compute_tilt():
    global Ax,Ay,Az
    L=np.linalg.norm([Ax,Ay,Az])
    ax,ay,az = Ax/L,Ay/L,Az/L
    rx=ax
    ry=az
    rz=-ay
    roll = math.atan2(ry,rz) #alpha
    pitch = math.asin(rx)#beta
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    return Ry @ Rx  # first roll, then pitch

            
class TemporalFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

class JointArrayPublisher(Node):
    def __init__(self):
        super().__init__('joint_array_publisher')
        self.publisher = self.create_publisher(Float32MultiArray, 'joint_array', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'joint_markers', 10)

    def publish_joint_array(self, joint_array):
        clean_array = np.nan_to_num(joint_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        msg = Float32MultiArray()
        msg.data = clean_array.flatten().tolist()
        self.publisher.publish(msg)

    def to_point(self, xyz):
        pt = Point()
        pt.x = xyz[0] / 1000.0
        pt.y = xyz[1] / 1000.0
        pt.z = xyz[2] / 1000.0
        return pt

    def publish_markers(self, joint_xyz):
        marker_array = MarkerArray()
        frame_id = "base"  # Adjust based on your setup
        
        # Publish joints as spheres
        for idx, (x, y, z) in enumerate(joint_xyz):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "joints"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x / 1000.0
            marker.pose.position.y = y / 1000.0
            marker.pose.position.z = z / 1000.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        # Publish bones (links) as lines
        connections = [
            ('CLAV', 'C7'), ('C7', 'LSHO'), ('C7', 'RSHO'),
            ('LSHO', 'LAEL'), ('LAEL', 'LWPS'),
            ('RSHO', 'RAEL'), ('RAEL', 'RWPS'),
            ('C7', 'L3'), ('L3', 'LHIP'), ('L3', 'RHIP'),
            ('LHIP', 'LKNE'), ('LKNE', 'LHEE'),
            ('RHIP', 'RKNE'), ('RKNE', 'RHEE')
        ]
        name_to_idx = {name: idx for idx, name in enumerate(JOINT_NAMES)}

        link_marker = Marker()
        link_marker.header.frame_id = frame_id
        link_marker.header.stamp = self.get_clock().now().to_msg()
        link_marker.ns = "links"
        link_marker.id = 1000
        link_marker.type = Marker.LINE_LIST
        link_marker.action = Marker.ADD
        link_marker.scale.x = 0.02
        link_marker.color.r = 0.0
        link_marker.color.g = 1.0
        link_marker.color.b = 0.0
        link_marker.color.a = 1.0

        for joint1, joint2 in connections:
            if joint1 in name_to_idx and joint2 in name_to_idx:
                p1 = joint_xyz[name_to_idx[joint1]]
                p2 = joint_xyz[name_to_idx[joint2]]
                if np.allclose(p1, 0) or np.allclose(p2, 0):
                    continue
                link_marker.points.append(self.to_point(p1))
                link_marker.points.append(self.to_point(p2))
        
        marker_array.markers.append(link_marker)
        self.marker_pub.publish(marker_array)


def deproject_pixel_to_point(u, v, depth, fx, fy, cx, cy,R):
    """
    Converts 2D pixel (u, v) and depth to 3D real-world coordinates in mm.
    """
    X = (u - cx) * depth / fx
    Z = (cy - v) * depth / fy
    Y = depth
    # print("depth ", Y)
    P = np.array([X, Y, Z]) 
    P = R@P
    P[1]-=1000
    P[2]+=800
    # return (X, Y, Z)
    return tuple(P)

def extract_joint_positions(results, depth_data):
    joint_depths = {}
    fx, fy = 997.117, 996.644
    cx, cy = 644.48, 463.817

    if not results.pose_landmarks:
        return joint_depths

    lsho = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rsho = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    mid_x = (lsho.x + rsho.x) / 2
    mid_y = (lsho.y + rsho.y) / 2
    c7_offset = -0.03
    R = compute_tilt()
    for name, offset in [('CLAV', 0.0), ('C7', c7_offset)]:
        u = int(mid_x * depth_data.shape[1])
        v = int((mid_y + offset) * depth_data.shape[0])
        if 0 <= u < depth_data.shape[1] and 0 <= v < depth_data.shape[0]:
            Z = depth_data[v, u]
            if Z == 0.0:
                neighborhood = depth_data[max(0, v - 20):min(v + 20, depth_data.shape[0]),
                                          max(0, u - 20):min(u + 20, depth_data.shape[1])].flatten()
                nonzero = neighborhood[neighborhood != 0]
                if len(nonzero):
                    Z = np.mean(nonzero)
            joint_depths[name] = deproject_pixel_to_point(u, v, Z, fx, fy, cx, cy,R)

    lhip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    rhip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    hip_x = (lhip.x + rhip.x) / 2
    hip_y = (lhip.y + rhip.y) / 2
    l3_x = mid_x + (hip_x - mid_x) * (2 / 3)
    l3_y = (mid_y + c7_offset) + (hip_y - (mid_y + c7_offset)) * (2 / 3)
    u = int(l3_x * depth_data.shape[1])
    v = int(l3_y * depth_data.shape[0])
    if 0 <= u < depth_data.shape[1] and 0 <= v < depth_data.shape[0]:
        Z = depth_data[v, u]
        if Z == 0:
            neighborhood = depth_data[max(0, v - 20):min(v + 20, depth_data.shape[0]),
                                      max(0, u - 20):min(u + 20, depth_data.shape[1])].flatten()
            nonzero = neighborhood[neighborhood != 0]
            if len(nonzero):
                Z = np.mean(nonzero)
        joint_depths['L3'] = deproject_pixel_to_point(u, v, Z, fx, fy, cx, cy,R)

    mp_joints = {
        'RSHO': mp_pose.PoseLandmark.RIGHT_SHOULDER,
        'LSHO': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'LAEL': mp_pose.PoseLandmark.LEFT_ELBOW,
        'RAEL': mp_pose.PoseLandmark.RIGHT_ELBOW,
        'LWPS': mp_pose.PoseLandmark.LEFT_WRIST,
        'RWPS': mp_pose.PoseLandmark.RIGHT_WRIST,
        'LHIP': mp_pose.PoseLandmark.LEFT_HIP,
        'RHIP': mp_pose.PoseLandmark.RIGHT_HIP,
        'LKNE': mp_pose.PoseLandmark.LEFT_KNEE,
        'RKNE': mp_pose.PoseLandmark.RIGHT_KNEE,
        'LHEE': mp_pose.PoseLandmark.LEFT_HEEL,
        'RHEE': mp_pose.PoseLandmark.RIGHT_HEEL
    }

    for name, idx in mp_joints.items():
        lm = results.pose_landmarks.landmark[idx]
        u, v = int(lm.x * depth_data.shape[1]), int(lm.y * depth_data.shape[0])
        if 0 <= u < depth_data.shape[1] and 0 <= v < depth_data.shape[0]:
            Z = depth_data[v, u]
            if Z == 0:
                neighborhood = depth_data[max(0, v - 50):min(v + 50, depth_data.shape[0]),
                                          max(0, u - 50):min(u + 50, depth_data.shape[1])].flatten()
                nonzero = neighborhood[neighborhood != 0]
                if len(nonzero):
                    Z = np.mean(nonzero)
            joint_depths[name] = deproject_pixel_to_point(u, v, Z, fx, fy, cx, cy,R)
    
    
    

    return {name: joint_depths.get(name, (0, 0, 0)) for name in JOINT_NAMES}

def main():
    rclpy.init()
    ros_node = JointArrayPublisher()

    config = Config()
    pipeline = Pipeline()
    temporal_filter = TemporalFilter(alpha=0.5)

    try:
        device = pipeline.get_device()
        pid = device.get_device_info().get_pid()
        
        
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(colorx, colory, OBFormat.MJPG, 30)
        config.enable_stream(color_profile)

        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(512, 512, OBFormat.Y16, 30)
        config.enable_stream(depth_profile)
        
        sensor_list:SensorList = device.get_sensor_list()
        accel_sensor = sensor_list.get_sensor_by_type(OBSensorType.ACCEL_SENSOR)
        accel_profile_list: StreamProfileList = accel_sensor.get_stream_profile_list()
        accel_profile: StreamProfile = accel_profile_list.get_stream_profile_by_index(0)
        assert accel_profile is not None
        accel_sensor.start(accel_profile, on_accel_frame_callback)

        config.set_align_mode(OBAlignMode.SW_MODE if pid == 0x066B else OBAlignMode.HW_MODE)
        pipeline.enable_frame_sync()
        pipeline.start(config)

        print("Orbbec Camera started. Press 'Q' to exit.")
    except Exception as e:
        print(f"Error initializing Orbbec: {e}")
        return

    frame_buffer = []

    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            continue

        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            continue

        width, height = depth_frame.get_width(), depth_frame.get_height()
        scale = depth_frame.get_depth_scale()

        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width)).astype(np.float32)
        depth_data *= scale
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0).astype(np.uint16)
    
        


        results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        joint_depths = extract_joint_positions(results, depth_data)
        
        
        joint_xyz = [joint_depths[name] for name in JOINT_NAMES]
        
        
        frame_buffer.append(joint_xyz)
        if len(frame_buffer) > 1:
            frame_buffer.pop(0)

        if len(frame_buffer) >= 1:
            joint_array = np.array(frame_buffer[-1:])  # number of frames
            ros_node.publish_joint_array(joint_array)
            ros_node.publish_markers(joint_xyz)

            # The `print` function in the code is used to display information or messages to the
            # console during the execution of the program. It is commonly used for debugging purposes
            # or to provide feedback to the user about the program's progress or state. In this
            # specific code snippet, the `print` statements are used to output messages such as
            # errors, status updates, and information about the joint array shape being published.
            # print("Published joint_array shape:", joint_array.shape)
        depth_vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        depth_resized = cv2.resize(depth_colored, (color_image.shape[1], color_image.shape[0]))

        combined_image = np.hstack((color_image, depth_resized))
        display_size = (int(combined_image.shape[1] * DISPLAY_SCALE), int(combined_image.shape[0] * DISPLAY_SCALE))
        resized_display = cv2.resize(combined_image, display_size)

        cv2.imshow("KIT Joint Tracker: RGB + Depth", resized_display)
        
        if cv2.waitKey(1) in [ord('q'), ESC_KEY]:
            break

    pipeline.stop()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

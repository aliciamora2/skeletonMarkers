import rospy
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from typing import List, Mapping, Optional, Tuple, Union
import time
import math
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from sensor_msgs import point_cloud2
from std_msgs.msg import Header, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Vector3
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import numpy as np
import ros_numpy
import statistics as st
import pyrealsense2 as rs2
import pandas as pd

bridge = CvBridge()

# Initialize MediaPipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_pose.Pose.static_image_mode=True,
mp_pose.Pose.model_complexity=2,
mp_pose.Pose.enable_segmentation=True,
mp_pose.Pose.min_detection_confidence=0.5
pose=mp_pose.Pose()

#---------------------------RVIZ PUBLISHER---------------------------
connections = [(11,13),(13,15),(15,17),(17,19),(19,21),(12,14),(14,16),(16,18),(18,20),(20,22),(11,23),(23,25),(25,27),(27,29),(29,31),(12,24),(24,26),(26,28),(28,30),(30,32),(11,12),(23,24)]
def publish_skeleton_joints(skeleton_data):
    marker = Marker()
    marker.header.frame_id = "camera_color_optical_frame"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "skeleton_joints"
    marker.id = 0
    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.1 # Size of the spheres
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.8
    marker.color.b = 0.46

    for point in skeleton_data:
        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = point[2]
        marker.points.append(p)
        
    marker2 = Marker()
    marker2.header.frame_id = "camera_color_optical_frame"
    marker2.header.stamp = rospy.Time.now()
    marker2.ns = "skeleton_joints"
    marker2.id = 1
    marker2.type = Marker.LINE_LIST
    marker2.action = Marker.ADD
    marker2.scale.x = 0.04 # Size of the lines
    marker2.color.a = 1.0
    marker2.color.r = 1.0
    marker2.color.g = 0.8
    marker2.color.b = 0.46

    for connection in connections:
        p1 = Point()
        p2 = Point()
        p1.x, p1.y, p1.z = skeleton_data[connection[0]]
        p2.x, p2.y, p2.z = skeleton_data[connection[1]]
        marker2.points.append(p1)
        marker2.points.append(p2)

    marker_array = MarkerArray()
    marker_array.markers.append(marker)
    marker_array.markers.append(marker2)

    marker_pub.publish(marker_array)
    rate.sleep()

#----------------------------MAIN PROGRAM----------------------------
class humanPose:
    def __init__(self):
        self.image_sub = Subscriber("camera/color/image_raw", Image)
        self.cloud_sub = Subscriber("camera/depth_registered/points", PointCloud2)
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.cloud_sub], 10, 0.01) # Subscribed topics are synchonized
        self.sync.registerCallback(self.callback)

    def callback(self, image_msg, cloud_msg):
    
        # Convert the PointCloud2 message to an array
        pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud_msg, remove_nans=False)
        pc_reshaped = np.reshape(pc, (-1, 3))

        # Convert the frame to RGB
        image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        imagen = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = imagen.shape

        # Process the frame with MediaPipe Pose
        result = pose.process(imagen)
    
        # Publish the 3D skeleton
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(imagen, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
            compute_skeleton = 0
        
            # Select pixel corresponding to the center of the hips
            right_hip = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * w , result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * h]
            left_hip = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * w , result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * h]
            hip = [int(np.round((right_hip[0]+left_hip[0])/2)), int(np.round((right_hip[1]+left_hip[1])/2))]
            
            if hip[0]<0 or hip[0]>=w or hip[1]<0 or hip[1]>=h: # Hips out of image bounds - try with shoulders
                # Select pixel corresponding to the center of the shoulders
                right_shoulder = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w , result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]
                left_shoulder = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w , result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
                shoulder = [int(np.round((right_shoulder[0]+left_shoulder[0])/2)), int(np.round((right_shoulder[1]+left_shoulder[1])/2))]
                if shoulder[0]>=0 and shoulder[0]<w and shoulder[1]>=0 and shoulder[1]<h: # Shoulders inside image bounds - skip skeleton if not
                    hip = shoulder
                    compute_skeleton = 1
            else:
                compute_skeleton = 1
 
            if compute_skeleton==1: # The skeleton is only published if depth information is available
                X=[]
                Y=[]
                Z=[]
            
                kernel_size = 5
                half_kernel = kernel_size // 2  # Calculate half of the kernel size

                for i in range(hip[1] - half_kernel, hip[1] + half_kernel + 1):
                    for j in range(hip[0] - half_kernel, hip[0] + half_kernel + 1):
                        x, y, z = pc[i][j]
                        X.append(x)
                        Y.append(y)
                        Z.append(z)

                hip_pc = [st.median(X), st.median(Y), st.median(Z)] #3D position of the hips
            
                landmark_points = []
                for index, landmark in enumerate(result.pose_world_landmarks.landmark): # Save transformed skeleton points in the landmark_points variable
                    landmark_points.append([hip_pc[0]+landmark.x, hip_pc[1]+landmark.y, hip_pc[2]+landmark.z])

                if not np.isnan(landmark_points).any():

                    publish_skeleton_joints(landmark_points)

if __name__ == '__main__':
    rospy.init_node("humanCapture")
    marker_pub = rospy.Publisher("skeleton_markers", MarkerArray, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    human_pose = humanPose()
    rospy.spin()


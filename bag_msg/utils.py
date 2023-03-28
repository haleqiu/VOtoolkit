import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
import numpy as np
from os.path import isfile, join, dirname, isdir
from os import listdir, mkdir, walk
from glob import glob as glob


def read_bag_message(bag_path):
    print(bag_path)
    bag = rosbag.Bag(bag_path, 'r')
    topics = bag.get_type_and_topic_info()[1].keys()
    print(topics)

def read_data_type(bag_path, tlist):
    bag = rosbag.Bag(bag_path, 'r')
    types = []
    for topic in tlist:
        types.append(bag.get_type_and_topic_info()[1][topic][0])
    print(types)
    
def read_pose(bag_path, t_list =['/integrated_to_init/'] , skip = 1):
    time_stamp = []
    pose = []
    bag = rosbag.Bag(bag_path, 'r')
    print(t_list)

    for topic, msg, t in bag.read_messages(topics=t_list):
        pose.append(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, 
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w], dtype=np.float64))
        time_stamp.append(np.array(msg.header.stamp.secs, dtype=np.float64) + np.array(1.0e-9 * msg.header.stamp.nsecs, dtype=np.float64)) # raw time 
    bag.close()
    return np.array(time_stamp)[:,None], np.array(pose)

def read_imu(bag_path, t_list =['/imu/data/'] , skip = 1):
    time_stamp = []
    orientation = []
    angular_velocity = []
    acceleration = []
    bag = rosbag.Bag(bag_path, 'r')
    print(t_list)
    for topic, msg, t in bag.read_messages(topics=t_list):
        orientation.append(np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w], dtype=np.float64))
        acceleration.append(np.array([msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z], dtype=np.float64))
        angular_velocity.append(np.array([msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z], dtype=np.float64))
        time_stamp.append(np.array(msg.header.stamp.secs, dtype=np.float64) + np.array(1.0e-9 * msg.header.stamp.nsecs, dtype=np.float64)) # raw time 
    bag.close()
    return np.array(time_stamp)[:,None], np.array(orientation), np.array(acceleration), np.array(angular_velocity)
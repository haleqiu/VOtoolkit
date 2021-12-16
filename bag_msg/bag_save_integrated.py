## NOTE this file should run with 

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
import numpy as np
from os.path import isfile, join, dirname, isdir
from os import listdir, mkdir, walk
from glob import glob as glob
import argparse


# Arguements
parser = argparse.ArgumentParser(description='save ros bag')
parser.add_argument("--topics", type=str, default=['/cam_image/2/depthplanner'], help="topic", nargs= "+")## multiple
parser.add_argument("--outdir", type=str, default=None, help="where to save the txt")
parser.add_argument("--inputdir", type=str, help="the folder for input bag file")
args = parser.parse_args(); print(args)

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
    
def read_integrated(bag_path, t_list =['/integrated_to_init/'] , skip = 1):
    time_stamp = []
    pose = []
    bag = rosbag.Bag(bag_path, 'r')
    for topic, msg, t in bag.read_messages(topics=t_list):
        
        pose.append(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, 
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]))
        time_stamp.append((msg.header.stamp.secs) + (1.0e-9 * msg.header.stamp.nsecs)) # raw time 
    bag.close()
    return np.array(time_stamp), np.array(pose)

def save_(save_obj, file_path):
    with open(file_path, "w") as time_txt:
        for o in save_obj:
            time_txt.write(o + "\n")

if __name__ == "__main__":
    input_bag_files = glob(args.inputdir + "/*.bag")
    for filename in input_bag_files:
        local_path = join(args.inputdir,filename.split('.')[0])
        if not isdir(local_path):
            mkdir(local_path)
        if args.outdir is None:
            outputdir = join(local_path, "Integrated")
        print(outputdir)
        if not isdir(outputdir):
            mkdir(outputdir)
        t, pose = read_integrated(filename)
        np.savetxt(join(outputdir, "time_stamp.txt"),t * 1.0e6)
        np.savetxt(join(outputdir, "pose.txt"),pose)

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
    
def read_imu(bag_path, outfolder, t_list =['/imu/data/'] , skip = 1):
    time_stamp = []
    orientation = []
    angular_velocity = []
    acceleration = []
    bag = rosbag.Bag(bag_path, 'r')
    for topic, msg, t in bag.read_messages(topics=t_list):
        orientation.append(np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]))
        acceleration.append(np.array([msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z]))
        angular_velocity.append(np.array([msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z]))
        time_stamp.append((msg.header.stamp.secs) + (1.0e-9 * msg.header.stamp.nsecs)) # raw time 
    bag.close()
    return np.array(time_stamp), np.array(orientation), np.array(acceleration), np.array(angular_velocity)

def save_(save_obj, file_path):
    with open(file_path, "w") as time_txt:
        for o in save_obj:
            time_txt.write(o + "\n")

if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='save ros bag')
    parser.add_argument("--topics", type=str, default=['/cam_image/2/depthplanner'], help="topic", nargs= "+")## multiple
    parser.add_argument("--outdir", type=str, default=None, help="where to save the txt")
    parser.add_argument("--inputdir", type=str, help="the folder for input bag file")
    args = parser.parse_args(); print(args)

    input_bag_files = glob(args.inputdir + "/*.bag")
    for filename in input_bag_files:
        local_path = join(args.inputdir,filename.split('.')[0])
        if not isdir(local_path):
            mkdir(local_path)
        if args.outdir is None:
            outputdir = join(local_path, "IMU")
        print(outputdir)
        if not isdir(outputdir):
            mkdir(outputdir)
        # read_bag_message(filename)
        # read_data_type(filename, ["/imu/data"])
        t, orin, acc, angular = read_imu(filename, None)
        np.savetxt(join(outputdir, "time_stamp.txt"),t * 1.0e6)
        np.savetxt(join(outputdir, "orientation.txt"),orin)
        np.savetxt(join(outputdir, "acceleration.txt"),acc)
        np.savetxt(join(outputdir, "angular_velocity.txt"),angular)

        #with open(join(outputdir,"rawtimes.txt"), "w") as time_txt:
        #    for time in time_stamp:
        #        time_txt.write(time + "\n")

## NOTE this file should run with 

# import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
import numpy as np
from os.path import isfile, join, dirname, isdir
from os import listdir, mkdir, walk
from glob import glob as glob
import argparse
import pickle
from utils import read_imu


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
    

def save_(save_obj, file_path):
    with open(file_path, "w") as time_txt:
        for o in save_obj:
            time_txt.write(o + "\n")

if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='save ros bag')
    parser.add_argument("--topics", type=str, default=['/imu/data'], help="topic", nargs= "+")## multiple
    parser.add_argument("--outdir", type=str, default=None, help="where to save the txt")
    parser.add_argument("--inputdir", type=str, help="the folder for input bag file")
    args = parser.parse_args(); print(args)

    input_bag_files = glob(args.inputdir + "/*/*.bag")
    print(args.inputdir + "/*/*.bag")
    print(input_bag_files)
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
        data = {
            'time_stamp':t,
            'orientation':orin,
            'acc':acc,
            'gyro':angular, 
        }
        with open(join(outputdir, "data.pkl"), 'wb') as file:
            pickle.dump(data,file, protocol=pickle.HIGHEST_PROTOCOL)
            print("save datapkl ", join(outputdir, "data.pkl"))

        

        #with open(join(outputdir,"rawtimes.txt"), "w") as time_txt:
        #    for time in time_stamp:
        #        time_txt.write(time + "\n")

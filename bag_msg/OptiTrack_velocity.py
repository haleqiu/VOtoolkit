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
from bag_save_integrated import read_integrated
from bag_save_IMU import read_imu, read_bag_message


def save_(save_obj, file_path):
    with open(file_path, "w") as time_txt:
        for o in save_obj:
            time_txt.write(o + "\n")

def read_velocity(bag_path, t_list =['/integrated_to_init/'] , skip = 1):
    time_stamp = []
    velocity = []
    bag = rosbag.Bag(bag_path, 'r')
    for topic, msg, t in bag.read_messages(topics=t_list):
        print(msg)
        velocity.append(np.array([msg.x, msg.y, msg.z]))
    bag.close()
    return np.array(velocity)

def read_pose(bag_path, t_list =['/integrated_to_init/'] , skip = 1):
    time_stamp = []
    pose = []
    bag = rosbag.Bag(bag_path, 'r')
    for topic, msg, t in bag.read_messages(topics=t_list):
        print(msg)
        pose.append(np.array([msg.x, msg.y, msg.z]))
    bag.close()
    return np.array(pose)

if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='save ros bag')
    parser.add_argument("--outdir", type=str, default=None, help="where to save the txt")
    parser.add_argument("--inputdir", type=str, help="the folder for input bag file")
    args = parser.parse_args(); print(args)
    input_bag_files = glob(args.inputdir + "/*.bag")
    for file_path in input_bag_files:
        
        ### check if output to the local directory
        filename = file_path.split('.')[0].split("/")[-1]
        if not args.outdir:
            local_path = join(args.inputdir,filename)
        else:
            local_path = join(args.outdir,filename)
        
        if not isdir(local_path):
            mkdir(local_path)
        outputdir = local_path
        print("output directory: ", outputdir)
        if not isdir(outputdir):
            mkdir(outputdir)

        read_bag_message(file_path)
        velocity = read_velocity(file_path, "/Robot_6/velocity_jenny")
        np.savetxt(join(outputdir, "velocity.txt"),velocity)

        pose = read_velocity(file_path, "/Robot_6/raw_pose")
        np.savetxt(join(outputdir, "pose.txt"),velocity)

        eular = read_velocity(file_path, "/Robot_6/eul_deg_jenny")
        np.savetxt(join(outputdir, "eular.txt"),eular)

        ## dataindexing
        # end_index = t_int.shape[0]
        # t_int, pose = t_int[integrate_index:end_index], pose[integrate_index:end_index]

        # print("time for imu %f, time for integrated %f"%(t_imu[0],t_int[0]))

        # with h5py.File(join(outputdir, "data.hdf5"), "w") as f:
        #     ts_s = f.create_dataset("ts", data=t_int)
        #     accel_dcalibrated_s = f.create_dataset("accel_dcalibrated", data=acc)
        #     gyro_dcalibrated_s = f.create_dataset("gyro_dcalibrated", data=angular)
        #     gyro_dcalibrated_s = f.create_dataset("gyro_dcalibrated", data=angular)
        #     vio_q_wxyz_s = f.create_dataset("integrated_q_wxyz", data=pose[:,3:])
        #     vio_p_s = f.create_dataset("integrated_p", data=pose[:,:3])

            


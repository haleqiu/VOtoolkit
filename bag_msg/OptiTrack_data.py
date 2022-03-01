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

def read_optitrack(bag_path, t_list =['/integrated_to_init/'] , skip = 1):
    time_stamp = []
    pose = []
    bag = rosbag.Bag(bag_path, 'r')
    for topic, msg, t in bag.read_messages(topics=t_list):
        print(msg)
        pose.append(np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, 
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]))
        time_stamp.append((msg.header.stamp.secs) + (1.0e-9 * msg.header.stamp.nsecs)) # raw time 
    bag.close()
    return np.array(time_stamp), np.array(pose)

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

        t_imu, orin, acc, angular = read_imu(file_path, None)
        t_opti, pose = read_optitrack(file_path, "/mocap_node/Robot_6/pose")
        read_bag_message(file_path)

        ## dataindexing
        # integrate_index = 1
        integrate_index = 0
        print(t_imu.shape)
        print(t_imu[0])
        # end_index = t_int.shape[0]
        end_index = t_imu.shape[0]
        # t_int, pose = t_int[integrate_index:end_index], pose[integrate_index:end_index]
        acc, angular, orin = acc[:end_index], angular[:end_index], orin[:end_index]

        np.savetxt(join(outputdir, "time_stamp.txt"),t_imu * 1.0e6)
        np.savetxt(join(outputdir, "orientation.txt"),orin)
        np.savetxt(join(outputdir, "acceleration.txt"),acc)
        np.savetxt(join(outputdir, "angular_velocity.txt"),angular)
        np.savetxt(join(outputdir, "opti_time_stamp.txt"),t_opti * 1.0e6)
        print(pose[0])
        np.savetxt(join(outputdir, "pose.txt"), pose)

        # print("time for imu %f, time for integrated %f"%(t_imu[0],t_int[0]))

        # with h5py.File(join(outputdir, "data.hdf5"), "w") as f:
        #     ts_s = f.create_dataset("ts", data=t_int)
        #     accel_dcalibrated_s = f.create_dataset("accel_dcalibrated", data=acc)
        #     gyro_dcalibrated_s = f.create_dataset("gyro_dcalibrated", data=angular)
        #     gyro_dcalibrated_s = f.create_dataset("gyro_dcalibrated", data=angular)
        #     vio_q_wxyz_s = f.create_dataset("integrated_q_wxyz", data=pose[:,3:])
        #     vio_p_s = f.create_dataset("integrated_p", data=pose[:,:3])

            


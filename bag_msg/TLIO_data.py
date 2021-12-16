## NOTE this file should run with 

import cv2
import rospy
import h5py
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
import numpy as np
from os.path import isfile, join, dirname, isdir
from os import listdir, mkdir, walk
from glob import glob as glob
import argparse
from bag_save_integrated import read_integrated
from bag_save_IMU import read_imu

# Arguements
parser = argparse.ArgumentParser(description='save ros bag')
parser.add_argument("--topics", type=str, default=['/cam_image/2/depthplanner'], help="topic", nargs= "+")## multiple
parser.add_argument("--outdir", type=str, default=None, help="where to save the txt")
parser.add_argument("--inputdir", type=str, help="the folder for input bag file")
args = parser.parse_args(); print(args)


def save_(save_obj, file_path):
    with open(file_path, "w") as time_txt:
        for o in save_obj:
            time_txt.write(o + "\n")

if __name__ == "__main__":
    input_bag_files = glob(args.inputdir + "/*.bag")
    for filename in input_bag_files:

        ### check if output to the local directory
        if not args.outdir:
            local_path = join(args.inputdir,filename.split('.')[0])
            if not isdir(local_path):
                mkdir(local_path)
            outputdir = local_path
        else:
            outputdir = args.outdir
        print("output directory: "outputdir)
        if not isdir(outputdir):
            mkdir(outputdir)

        t_int, pose = read_integrated(filename)
        t_imu, orin, acc, angular = read_imu(filename, None)

        ## dataindexing
        integrate_index = 1
        print(t_imu.shape)
        print(t_int.shape)
        print(t_imu[0])
        print(t_int[0])
        end_index = t_int.shape[0]
        t_int, pose = t_int[integrate_index:end_index], pose[integrate_index:end_index]
        acc, angular, orin = acc[:end_index], angular[:end_index], orin[:end_index]

        print("time for imu %f, time for integrated %f"%(t_imu[0],t_int[0]))

        with h5py.File(join(outputdir, "data.hdf5"), "w") as f:
            ts_s = f.create_dataset("ts", data=t_int)
            accel_dcalibrated_s = f.create_dataset("accel_dcalibrated", data=acc)
            gyro_dcalibrated_s = f.create_dataset("gyro_dcalibrated", data=angular)
            gyro_dcalibrated_s = f.create_dataset("gyro_dcalibrated", data=angular)
            vio_q_wxyz_s = f.create_dataset("integrated_q_wxyz", data=pose[:,3:])
            vio_p_s = f.create_dataset("integrated_p", data=pose[:,:3])

            


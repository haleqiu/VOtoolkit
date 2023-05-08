import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag, os, argparse
import numpy as np
from os.path import isfile, join, dirname, isdir
from os import listdir, mkdir, walk
from glob import glob as glob
import pickle

from utils import read_pose, read_imu, read_twist
from utils import read_bag_message

if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='save ros bag')
    parser.add_argument("--topics", type=str, default=['/sensor_head/gpsins/pose'], help="topic", nargs= "+")## multiple
    parser.add_argument("--outdir", type=str, default=None, help="where to save the txt")
    parser.add_argument("--inputdir", type=str, help="the folder for input bag file")
    args = parser.parse_args(); print(args)

    input_bag_files = glob(args.inputdir + "/*.bag")
    for filename in input_bag_files:
        
        local_path = join(args.inputdir,filename.split('.')[0])
        os.makedirs(local_path, exist_ok=True)

        if args.outdir is None:
            outputdir = local_path
        print(outputdir)

        os.makedirs(outputdir, exist_ok=True)

        # t, pose = read_pose(filename, args.topics)
        # pose_data = np.concatenate([t, pose], axis=-1)
        # np.savetxt(join(outputdir, "pose.txt"),pose_data)
        # with open(join(outputdir, "pose.pkl"), 'wb') as handle:
        #     pickle.dump(pose_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        t, twist = read_twist(filename, args.topics)
        twist_data = np.concatenate([t, twist], axis=-1)
        np.savetxt(join(outputdir, "twist.txt"),twist_data)
        with open(join(outputdir, "twist.pkl"), 'wb') as handle:
            pickle.dump(twist_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        t, orin, acc, angular = read_imu(filename, ['/sensor_head/gpsins/imu'])

        imu_data = np.concatenate([t, acc, angular], axis=-1)
        np.savetxt(join(outputdir, "orientation.txt"),orin)
        np.savetxt(join(outputdir, "imu.txt"),imu_data)
        with open(join(outputdir, "imu.pkl"), 'wb') as handle:
            pickle.dump(imu_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

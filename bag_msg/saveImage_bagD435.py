import os
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
from os.path import isfile, join, dirname, isdir
from os import listdir, mkdir
import argparse
import glob, tqdm

from sensor_msgs.msg import CameraInfo

def camera_info_callback(msg):
    # This function is called whenever a new message is received on the topic
    rospy.loginfo("Received CameraInfo message")
    
    # Accessing the fields of the CameraInfo message
    rospy.loginfo(f"Header: {msg.header}")
    rospy.loginfo(f"Height: {msg.height}")
    rospy.loginfo(f"Width: {msg.width}")
    rospy.loginfo(f"Distortion model: {msg.distortion_model}")
    rospy.loginfo(f"D: {msg.D}")
    rospy.loginfo(f"K: {msg.K}")
    rospy.loginfo(f"R: {msg.R}")
    rospy.loginfo(f"P: {msg.P}")
    rospy.loginfo(f"binning_x: {msg.binning_x}")
    rospy.loginfo(f"binning_y: {msg.binning_y}")
    rospy.loginfo(f"roi: {msg.roi}")

# Arguements
parser = argparse.ArgumentParser(description='save ros bag')
parser.add_argument("--topics", type=str, default=['/device_0/sensor_0/Infrared_1/image/data', '/device_0/sensor_0/Infrared_2/image/data'], help="topic", nargs= "+")## multiple
parser.add_argument("--outdir", type=str, default='/home/yuheng/data/realsensedata', help="where to save the txt")
parser.add_argument("--inputdir", type=str, default='/home/yuheng/data/realsensebag', help="the folder for input bag file")
args = parser.parse_args(); print(args)

# outvidfile = 'bag_save6.avi'
skip = 1
cvbridge = CvBridge()
SaveVideo = False
image_size = (960,600)


for filename in glob.glob(args.inputdir + "/*.bag"):
    print(filename)
    # save multiple bagfiles in one folder into one single video

    filepathname = join(args.inputdir, "bag", filename)

    local_path = join(args.outdir, filename.split('/')[-1])
    if not isdir(local_path):
        mkdir(local_path)
        mkdir(join(local_path, "rgb_l"))
        mkdir(join(local_path, "rgb_r"))

    if SaveVideo:
        outvidfile = subfolder+'.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fout=cv2.VideoWriter(outvidfile, fourcc, 30.0, image_size)

    bag = rosbag.Bag(filepathname, 'r')
    for subfolder, t in zip(["rgb_l", "rgb_r"], args.topics):
        os.makedirs(join(local_path, subfolder), exist_ok=True)
        print(t)
        ind = 0
        time_txt = open(join(local_path, subfolder, "times.txt"), "a")

        # try:
            # Initialize the ROS node
        rospy.init_node('camera_info_listener', anonymous=True)
        
        # Subscribe to the topic that publishes CameraInfo messages
        info_topic = t[:-10] + "info/camera_info"
        print(info_topic)
        rospy.Subscriber(info_topic, CameraInfo, camera_info_callback)
        
        # Keep the node running until it is shut down
        rospy.spin()
        
        for info in tqdm.tqdm(bag.read_messages(topics=t)):
            topic, msg, t = info
            if ind%skip==0:
                # print(f"Image encoding: {msg.encoding}")
                image_np = cvbridge.imgmsg_to_cv2(msg, "passthrough")  # TODO: check the encoding
                if SaveVideo:
                    fout.write(image_np)
                else:
                    imagename = "%06d"%ind+'.png'
                    cv2.imwrite(join(local_path,subfolder, imagename), image_np)
                    # print(join(local_path,subfolder, imagename))
                    time_stamp = str(msg.header.stamp.secs) + "." + str(msg.header.stamp.nsecs)
                    time_txt.write(time_stamp + "\n")
            ind = ind + 1
        time_txt.close()
    bag.close()

    if SaveVideo:
        fout.release()

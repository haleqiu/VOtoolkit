## NOTE this file should run with 

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
import numpy as np
import os
from os.path import join
from os import listdir
import argparse


def save_depth(bag_path, outfolder, t_list =['/cam_image/0/depthplanner'], SaveVideo = False, skip = 1):
    # This function is ran for single msg like the depth of the left camera.
    if SaveVideo:
        outvidfile = outfolder+'.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fout=cv2.VideoWriter(outvidfile, fourcc, 30.0, image_size)

    time_stamp = []
    bag = rosbag.Bag(bag_path, 'r')
    ind = 0
    for topic, msg, t in bag.read_messages(topics=t_list):
        if ind%skip==0:
            image_np = cvbridge.imgmsg_to_cv2(msg)  # TODO: check the encoding
            image_np = np.array(image_np,dtype=np.float32)
            time_stamp.append(str(msg.header.stamp.secs) + "." + str(msg.header.stamp.nsecs))#TODO neumeric
            if SaveVideo:
                fout.write(image_np)
            else:
                imagename = "%06d"%(ind)+".png"
                cv2.imwrite(join(outputdir,imagename), image_np)
                np.save(join(outputdir,"%06d"%(ind)),image_np)

        ind = ind + 1
    bag.close()

    if SaveVideo:
        fout.release()
    return time_stamp


if __name__ == "__main__":

    # Arguements
    parser = argparse.ArgumentParser(description='save ros bag')
    parser.add_argument("--topics", type=str, default=['/cam_image/2/depthplanner', '/cam_image/1/depthplanner'], help="topic", nargs= "+")## multiple
    parser.add_argument("--outdir", type=str, default=None, help="where to save the txt")
    parser.add_argument("--inputdir", type=str, default='/data/dynamic_person/', help="the folder for input bag file")
    args = parser.parse_args(); print(args)

    cvbridge = CvBridge()
    SaveVideo = False
    image_size = (360,640)
    for filename in os.listdir(join(args.inputdir, "bag")):
        local_path = join(args.inputdir,filename.split('.')[0])
        if not args.outdir:
            outputdir = join(local_path, "depth")
        else:
            outputdir = join(args.outdir, "depth")

        print(outputdir)
        os.makedirs(outputdir, exist_ok = True)

        bag_path = join(join(args.inputdir,"bag"),filename)
        time_stamp = save_depth(bag_path,outputdir,t_list=[args.topics[0]])
        with open(join(outputdir,"rawtimes0.txt"), "w") as time_txt:
            for time in time_stamp:
                time_txt.write(time + "\n")
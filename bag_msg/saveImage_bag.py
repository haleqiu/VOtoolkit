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

# Arguements
parser = argparse.ArgumentParser(description='save ros bag')
parser.add_argument("--topics", type=str, default=['/zedx/zed_node/left/image_rect_color', '/zedx/zed_node/right/image_rect_color'], help="topic", nargs= "+")## multiple
parser.add_argument("--outdir", type=str, default='', help="where to save the txt")
parser.add_argument("--inputdir", type=str, default='/data2/yuhengq/dsta_payload/2024_08_09/run2/zed_bags', help="the folder for input bag file")
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
        
        for info in tqdm.tqdm(bag.read_messages(topics=t)):
            topic, msg, t = info
            if ind%skip==0:
                # print(f"Image encoding: {msg.encoding}")
                image_np = cvbridge.imgmsg_to_cv2(msg, "passthrough")  # TODO: check the encoding
                if SaveVideo:
                    fout.write(image_np)
                else:
                    time_stamp = str(msg.header.stamp.secs) + "_" +  str(msg.header.stamp.nsecs)
                    imagename = time_stamp + '.png'
                    cv2.imwrite(join(local_path,subfolder, imagename), image_np)
                    time_txt.write(time_stamp + "\n")
            ind = ind + 1
        time_txt.close()
    bag.close()

    if SaveVideo:
        fout.release()

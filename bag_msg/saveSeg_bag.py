import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
from os.path import isfile, join, dirname, isdir
from os import listdir, mkdir
import argparse
import numpy as np

# Arguements
parser = argparse.ArgumentParser(description='save ros bag')
parser.add_argument("--topics", type=str, default=['/cam_image/1/segmentation', '/cam_image/2/segmentation'], help="topic", nargs= "+")## multiple
parser.add_argument("--outdir", type=str, default='/data/dynamic_person/', help="where to save the txt")
parser.add_argument("--inputdir", type=str, default='/data/dynamic_person/', help="the folder for input bag file")
parser.add_argument("--bag", type=str, default='/data/dynamic_person/bag/2020-08-16-23-33-27.bag', help="the folder for input bag file")
args = parser.parse_args(); print(args)


# outvidfile = 'bag_save6.avi'
skip = 1
cvbridge = CvBridge()
SaveVideo = False
image_size = (960,720)


for filename in listdir(join(args.inputdir, "bag")):
    print(filename)
    # save multiple bagfiles in one folder into one single video

    filepathname = join(args.inputdir, "bag", filename)
    if (not isfile(filepathname)) or (not filename[-3:]=='bag'):
        continue

    local_path = join(args.inputdir,filename.split('.')[0])
    if not isdir(local_path):
        mkdir(local_path)
    if not isdir(join(local_path, "image_seg_0")):
        mkdir(join(local_path, "image_seg_0"))
        mkdir(join(local_path, "image_seg_1"))

    if SaveVideo:
        outvidfile = subfolder+'.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fout=cv2.VideoWriter(outvidfile, fourcc, 30.0, image_size)

    bag = rosbag.Bag(filepathname, 'r')
    subfolder = "image_seg_0"## TODO cam1 is right image in airsim hhh
    for t in args.topics:
        print(t)
        ind = 0
        for topic, msg, t in bag.read_messages(topics=t):
            if ind%skip==0:
                image_np = cvbridge.imgmsg_to_cv2(msg, "rgb8")  # TODO: check the encoding
                if SaveVideo:
                    fout.write(image_np)
                else:
                    imagename = "%06d"%ind+'.png'
                    cv2.imwrite(join(local_path,subfolder, imagename), image_np)
                    # np.savetxt(join(local_path,subfolder,"%06d"%ind+'.txt'),image_np, fmt='%s')
            ind = ind + 1
        subfolder = "image_seg_1"
    bag.close()
    # print(image_np)

    if SaveVideo:
        fout.release()

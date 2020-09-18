# read drone pose from the bagfile and save them
import rospy
import rosbag
from os.path import isfile, join, isdir
from os import listdir, mkdir
import numpy as np
import json
import argparse

# Arguements
# TODO not consistent
parser = argparse.ArgumentParser(description='save ros bag')
parser.add_argument("--topics", type=str, default=['/cam_image/0/camera_pose'], help="topic", nargs= "+")## multiple
parser.add_argument("--outdir", type=str, default='/data/dynamic_person/txt', help="where to save the txt")
parser.add_argument("--inputdir", type=str, default='/data/dynamic_person/', help="the folder for input bag file")
args = parser.parse_args(); print(args)

np.set_printoptions(precision=3, suppress=True, threshold=10000)

topiclist = args.topics
if not isdir(args.outdir):
    mkdir(args.outdir)

for filename in listdir(join(args.inputdir, "bag")):
    filepathname = join(args.inputdir, "bag",filename)
    print(filepathname)

    if (not isfile(filepathname)) or (not filename[-3:]=='bag'):
        continue

    bag = rosbag.Bag(filepathname, 'r')
    topics = bag.read_messages(topics=topiclist)

    pose_np = []
    for topic, msg, t in topics:
        pos = msg.pose.position
        ori = msg.pose.orientation
        pose_np.append(np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]))

    if not isdir(args.outdir):
        mkdir(args.outdir)
    posefilename = join(args.outdir,filename.split('.')[0]+'_pose.txt')
    print(posefilename)
    np.savetxt(posefilename, np.array(pose_np))

    bag.close()

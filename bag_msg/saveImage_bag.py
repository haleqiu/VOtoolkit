import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
from os.path import isfile, join, dirname, isdir
from os import listdir, mkdir
import argparse

# Arguements
parser = argparse.ArgumentParser(description='save ros bag')
parser.add_argument("--topics", type=str, default=['/cam_image/1/scene', '/cam_image/2/scene'], help="topic", nargs= "+")## multiple
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
        mkdir(join(local_path, "image_0"))
        mkdir(join(local_path, "image_1"))

    if SaveVideo:
        outvidfile = subfolder+'.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fout=cv2.VideoWriter(outvidfile, fourcc, 30.0, image_size)

    bag = rosbag.Bag(filepathname, 'r')
    subfolder = "image_0"## TODO
    for t in args.topics:
        ind = 0
        time_txt = open(join(local_path, subfolder,"times.txt"), "a")
        for topic, msg, t in bag.read_messages(topics=t):
            if ind%skip==0:
                image_np = cvbridge.imgmsg_to_cv2(msg, "rgb8")  # TODO: check the encoding
                    # cv2.imshow('img', image_np)
                    # cv2.waitKey(0)
                    # image_np = cv2.resize(image_np, (640, 360)) # Uncomment if you want to resize the image
                    # print image_np.shape
                if SaveVideo:
                    fout.write(image_np)
                else:
                    imagename = "%06d"%ind+'.png'
                    cv2.imwrite(join(local_path,subfolder, imagename), image_np)
                    time_stamp = str(msg.header.stamp.secs) + "." + str(msg.header.stamp.nsecs)
                    time_txt.write(time_stamp + "\n")
            ind = ind + 1
        subfolder = "image_1"
        time_txt.close()
    bag.close()

    if SaveVideo:
        fout.release()

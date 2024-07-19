# undistort by OpenCV functions
import cv2
import numpy as np
import yaml
import os
import pypose as pp
import glob
import tqdm

import matplotlib.pyplot as plt

def recity_Stereo(map1_x, map1_y, map2_x, map2_y, img1, img2):

    rectified_img1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
    rectified_img2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)

    return rectified_img1, rectified_img2

def plot_rect_img(img1, img2):
    
    fig = plt.figure(figsize=(30, 10))
    cat = np.concatenate([img1, img2], axis=1)
    plt.imshow(cv2.cvtColor(cat, cv2.COLOR_BGR2RGB))

    for i in range(0, img1.shape[1], 100):
        plt.plot([0, 2*img1.shape[0]], [i, i], 'r-')

def to_K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def parse_intrinsics(cam_calibration):
    xi, fx, fy, cx, cy = cam_calibration["intrinsics"]
    dist = cam_calibration["distortion_coeffs"]

    return np.array(xi), to_K(fx, fy, cx, cy), np.array(dist)

def omin_stereo_rectify(left_image_folder, right_image_folder, left_params, right_params, TCN, out_folder):
    xil, Kl, Dl = parse_intrinsics(left_params)
    xir, Kr, Dr = parse_intrinsics(right_params)
    
    R = TCN[:3,:3]
    T = TCN[:3,3]

    RL, RR = cv2.omnidir.stereoRectify(R, T)
    
    image_size = (600, 600)
    
    P = np.array([
        [200, 0, 300],
        [0, 200, 300],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Undistort and rectify the images
    mapl_x, mapl_y = cv2.omnidir.initUndistortRectifyMap(
        Kl, Dl, xil, RL, P, image_size, cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
    )
    mapr_x, mapr_y = cv2.omnidir.initUndistortRectifyMap(
        Kr, Dr, xir, RR, P, image_size, cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
    )
    
    files_left = glob.glob(os.path.join(left_image_folder, "*.jpg"))
    files_right = glob.glob(os.path.join(right_image_folder, "*.jpg"))
    files_left.sort()
    files_right.sort()

    for file_left, file_right in tqdm.tqdm(zip(files_left, files_right)):
        imgl = cv2.imread(file_left)
        imgr = cv2.imread(file_right)

        rectified_imgl = cv2.remap(imgl, mapl_x, mapl_y, cv2.INTER_LINEAR)
        rectified_imgr = cv2.remap(imgr, mapr_x, mapr_y, cv2.INTER_LINEAR)

        # plot_rect_img(img1, img0)
        os.makedirs(os.path.join(out_folder, "rgb_l"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, "rgb_r"), exist_ok=True)
        # 0 is right camera
        cv2.imwrite(os.path.join(out_folder, "rgb_l", os.path.basename(file_left)), rectified_imgl)
        cv2.imwrite(os.path.join(out_folder, "rgb_r", os.path.basename(file_right)), rectified_imgr)


if __name__ == "__main__":
    base_folder = "/home/yuhneg/data/dsta_payload/07-17"
    
    
    # for cam 01
    out_folder = os.path.join(base_folder, "cam01")
    calibration_file = os.path.join(base_folder, "cam0-1.yaml")
    left_image_folder = os.path.join(base_folder, "frames_1")
    right_image_folder = os.path.join(base_folder, "frames_0")

    with open(calibration_file, "r") as f:
        params = yaml.safe_load(f)
        
    E = pp.mat2SE3(np.array(params["cam1"]["T_cn_cnm1"]))
    
    omin_stereo_rectify(left_image_folder, right_image_folder, left_params=params["cam1"], right_params=params["cam0"], TCN= E.Inv().matrix().numpy(), out_folder=out_folder)
    
    # for cam 45
    out_folder = os.path.join(base_folder, "cam45")
    calibration_file = os.path.join(base_folder, "cam4-5.yaml")
    left_image_folder = os.path.join(base_folder, "frames_4")
    right_image_folder = os.path.join(base_folder, "frames_5")

    with open(calibration_file, "r") as f:
        params = yaml.safe_load(f)
        
    E = pp.mat2SE3(np.array(params["cam1"]["T_cn_cnm1"]))
    
    omin_stereo_rectify(left_image_folder, right_image_folder, left_params=params["cam0"], right_params=params["cam1"], TCN= E.matrix().numpy(), out_folder=out_folder)


    
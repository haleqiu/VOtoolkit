import os
import cv2
import numpy as np
import yaml
import os.path as osp
import pypose as pp
import glob
import tqdm
import pandas as pd
import struct

import itertools
from tqdm import tqdm

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


# converted from rosbag_mp4 from cstacks bitbucket
def read_ros_time_timestamps(file_path):
    image_timestamps = []
    log_timestamps = []
    with open(file_path, 'rb') as file:
        # Read the output topic name until the newline character
        output_topic_name = ""
        while True:
            c = file.read(1).decode('utf-8')
            if c == '\n':
                break
            output_topic_name += c
        print(f"OUTPUT TOPIC NAME: {output_topic_name}")
        
        # Read the rest of the file as 32-bit unsigned integers
        file_content = file.read()
        num_elements = len(file_content) // 4  # Each uint32_t is 4 bytes
        
        # Unpack the data
        data = struct.unpack('I' * num_elements, file_content)
        
        # Convert the data to timestamps
        for i in range(0, num_elements, 4):
            secs = data[i]
            nsecs = data[i + 1]
            image_timestamp = secs + nsecs / 1e9

            secs = data[i + 2]
            nsecs = data[i + 3]
            log_timestamp = secs + nsecs / 1e9
            
            image_timestamps.append(image_timestamp)
            log_timestamps.append(log_timestamp)
            
    
    return np.array(image_timestamps), np.array(log_timestamps), data

def get_frame_count(file_path):
    """
    Get the frame count of a video file using mediainfo.

    :param file_path: Path to the video file.
    :return: Frame count as an integer, or None if an error occurs.
    """
    try:
        result = subprocess.run(
            ['mediainfo', '--Inform=Video;%FrameCount%', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            frame_count = int(result.stdout.strip())
            return frame_count
        else:
            print(f"Error: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def video_to_frames(video_path, output_dir, shift=0):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = -shift
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left to read

        # Save the frame as an image file
        if frame_count >= 0:
            frame_filename = f"{output_dir}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Done! Extracted {frame_count} frames.")

def try_index_shift(df : pd.DataFrame, index_shifts : list):
    
    df2 = df.copy()

    for i, row in df2.iterrows():

        # apply shift to timestamps and frame index
        
        shift = index_shifts[i]
        df2.at[i, "img_index"] = row["img_index"][shift:]
        df2.at[i, "img_ts_from_start"] = row["img_ts_from_start"][shift:]
        df2.at[i, "img_ts"] = row["img_ts"][shift:]
        df2.at[i, "frame_index"] = row["frame_index"][shift:]
        df2.at[i, "num_timestamps"] = row["num_timestamps"] - shift

    # obtain the shortest length after this index shift
    min_num_timestamps = df2["num_timestamps"].min()

    # truncate the data to the shortest length
    df2["img_index"] = df2["img_index"].apply(lambda x: x[:min_num_timestamps])
    df2["img_ts_from_start"] = df2["img_ts_from_start"].apply(lambda x: x[:min_num_timestamps])
    df2["img_ts"] = df2["img_ts"].apply(lambda x: x[:min_num_timestamps])
    df2["frame_index"] = df2["frame_index"].apply(lambda x: x[:min_num_timestamps])
    df2["num_timestamps"] = min_num_timestamps


    # calculate the mean time difference between frames in synchronized image frame
    all_ts = np.vstack(df2["img_ts_from_start"].tolist())

    ts_diff = np.max(all_ts, axis=0) - np.min(all_ts, axis=0)
    ts_diff_mean = np.mean(ts_diff)
    ts_diff_std = np.std(ts_diff)
    ts_diff_max = np.max(ts_diff)

    return ts_diff_mean, ts_diff_std, ts_diff_max, df2


if __name__ == "__main__":
    base_folder = "/data2/yuhengq/dsta_payload/2024_08_09/run1/videos"

    video_paths = glob.glob(osp.join(base_folder, "*.mp4"))
    video_paths.sort()

    df_data = []

    for video_path in video_paths:
        timestamps_path = video_path.replace(".mp4", ".timestamps")

        camera_name = osp.basename(video_path).split("_")[0]
        img_ts, log_ts, data = read_ros_time_timestamps(timestamps_path)

        mp4_frames = get_frame_count(video_path)

        df_data.append({
            "camera": camera_name,
            "video_path": video_path,
            "num_mp4_frames": mp4_frames,
            "img_ts": img_ts,
            "log_ts": log_ts,
            "num_timestamps": len(img_ts),
            "img_dt_mean" : np.mean(np.diff(img_ts)),
            "img_dt_std" : np.std(np.diff(img_ts)),
            "log_dt_mean" : np.mean(np.diff(log_ts)),
            "log_dt_std" : np.std(np.diff(log_ts)),
            "frame_index" : np.arange(len(img_ts))
        })

    df = pd.DataFrame(df_data)
    # Define the range for each element
    range_values = [0, 1, 2, 3]

    # Get all combinations where the length of the list is 6
    combinations = list(itertools.product(range_values, repeat=6))
    result = []

    #determine start and dt
    time_start = min(df["img_ts"].apply(lambda x: x[0]).tolist())

    df['img_ts_from_start'] = df['img_ts'] - time_start

    dt_unit = df["log_dt_mean"].mean()
    print("will resample image stream into dt_unit = ", dt_unit)

    # calculate integer index for image ts
    df['img_index'] = df['img_ts_from_start'].apply(lambda x: (x / dt_unit))

    for c in tqdm(combinations):
        ts_diff_mean, ts_diff_std, ts_diff_max, _ = try_index_shift(df, c)
        result.append({
            "index_shifts": c,
            "ts_diff_mean": ts_diff_mean,
            "ts_diff_std": ts_diff_std,
            "ts_diff_max": ts_diff_max
        })

    result_df = pd.DataFrame(result)
    r = result_df.sort_values("ts_diff_mean")
    shif_index = r.iloc[0]["index_shifts"]
    print(shif_index)

    video_to_frames(video_paths[0], osp.join(base_folder, "frames_0"), shift=shif_index[0])
    video_to_frames(video_paths[1], osp.join(base_folder, "frames_1"), shift=shif_index[1])
    video_to_frames(video_paths[2], osp.join(base_folder, "frames_2"), shift=shif_index[2])
    video_to_frames(video_paths[3], osp.join(base_folder, "frames_3"), shift=shif_index[3])
    video_to_frames(video_paths[4], osp.join(base_folder, "frames_4"), shift=shif_index[4])
    video_to_frames(video_paths[5], osp.join(base_folder, "frames_5"), shift=shif_index[5])
    
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


    
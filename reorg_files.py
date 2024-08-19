
import os
import shutil, glob
import tqdm
import argparse

import numpy as np

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--folder", type=str, default="/home/yuhneg/data/dsta_payload/07-19-run1")
    args = args.parse_args()
    
    out_dir = os.path.join(args.folder, "zed_07-19-run1")
    os.makedirs(out_dir, exist_ok=True)
    bag_folders =  glob.glob(args.folder + "/zed_*")
    bag_folders.sort()
    for zed_fold in bag_folders:
        if os.path.isdir(zed_fold):
            print(zed_fold)
            for subfold in ["rgb_l", "rgb_r"]:
                os.makedirs(os.path.join(out_dir, subfold), exist_ok=True)
                
                time_stamp = np.loadtxt(os.path.join(zed_fold, subfold, "times.txt"),dtype='str')
                image_files = glob.glob(os.path.join(zed_fold, subfold, "*.png"))
                assert time_stamp.shape[0] == len(image_files)
                for timestamp, image_path in tqdm.tqdm(zip(time_stamp, image_files)):
                    shutil.move(image_path, os.path.join(out_dir, subfold, timestamp + ".png"))
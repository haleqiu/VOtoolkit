
import os
import shutil, glob
import tqdm
import argparse
import pdb

import numpy as np

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--folder", type=str, default="/data2/yuhengq/dsta_payload/2024_08_09/run2/zed_bags")
    args = args.parse_args()
    
    out_dir = os.path.join(args.folder, "zed")
    os.makedirs(out_dir, exist_ok=True)
    bag_folders =  glob.glob(args.folder + "/zed_*")
    bag_folders.sort()
    for subfold in ["rgb_l", "rgb_r"]:
        idx = 0
        os.makedirs(os.path.join(out_dir, subfold), exist_ok=True)
        for zed_fold in bag_folders:
            if os.path.isdir(zed_fold):
                print(zed_fold)
                time_stamp = np.loadtxt(os.path.join(zed_fold, subfold, "times.txt"),dtype='str')
                image_files = glob.glob(os.path.join(zed_fold, subfold, "*.png"))
                image_files.sort()
                # assert time_stamp.shape[0] == len(image_files)
                for i, (timestamp, image_path) in enumerate(tqdm.tqdm(zip(time_stamp, image_files))):
                    # shutil.move(image_path, os.path.join(out_dir, subfold, timestamp + ".png"))
                    img_idx = idx + i 
                    shutil.move(image_path, os.path.join(out_dir, subfold,  "%06d.png"%img_idx))
                idx+=i
                print(bag_folders)
                print(idx)
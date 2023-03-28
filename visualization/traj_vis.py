
import datetime
import cv2
import imageio
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
import os
import pypose as pp



def create_gif_from_image_lists(image_lists, image_labels, gif_fname, fps=1):
    '''
    Create a gif from a list of images.
    Args:
        image_lists(list): List of images. Each entry is a (H, W, C) np array. The format is BGR. Pixels are [0, 255] uint8.
        image_labels(list): List of image labels.
        gif_name(str): Name of the gif.
        fps(int): Frames per second.
    '''

    # Create the output image. Concatenated horizontally.
    # A black bar.
    barwidth = 10
    black_bar = np.array([0, 0, 0] * image_lists[0][0].shape[0] * barwidth).reshape((-1, barwidth, 3))
    frames = []
    for i in range(len(image_lists[0])):
        frame_images = [l[i] for l in image_lists]

        # Add a bar between images.
        frame_images = [np.concatenate([f, black_bar], axis=1) for f in frame_images[:-1]] + [frame_images[-1]]

        frame = np.concatenate(frame_images, axis=1)

        # Frame to uint8. Range [0, 255].
        frame = (frame).astype(np.uint8)

        # Add the frame.
        frames.append(frame)

    # Create gif.
    imageio.mimsave(gif_fname, frames, fps=fps)


def create_trajectory_summary_video(results, image_lists, data_lists, data_names, output_fpath):
    '''
    Creates a visualization of the trajectory estimation result. Add as many image-lists as you want, and this script will also handle putting them together in a GIF, showing the i-th image of each list in the same frame.

    Args:
        results (dict): Dictionary containing the results of the trajectory estimation. Must contain the following keys:
            - est_traj (np.ndarray): Estimated trajectory. Shape: (N, 7), xyz, xyzw.
            - gt_traj (np.ndarray): Ground truth trajectory. Shape: (N, 7), xyz, xyzw. Realistically, only the xy portions are used here.

            These trajectories are in the world coordinate system, typically AFTER being scaled and rotated by an ATE estimation algorithm.
        image_lists (list): List of lists of images, with each having one image per frame. The images will be shown sequentially.
        data_lists (list): List of lists of data, with each having one datapoint per frame. The i-th datapoint of each data-list will be highlighted in the same time the frame is shown.
    '''
    ################################
    # Create GIF or video.
    ################################
    # Create images of the loss over samples, and highlight the current sample in each image.
    # Create a list of images.
    est_traj_xyz = results['est_traj']
    gt_traj_xyz = results['gt_traj']

    data_plot_image_list = []
    traj_img_list = []
    print("Creating GIF.")
    for i in tqdm(range(len(image_lists[0]))):

        # Create plot image for the input data lists.
        fig, axes = plt.subplots(ncols=1, nrows=len(data_lists))
        fig.set_size_inches(5, 5)

        for j, data_list in enumerate(data_lists):
            ax = axes[j]
            ax.plot(data_list)

            # Highlight the current frame.
            ax.plot(i, data_list[i], 'ro')

            # Axes naming.
            ax.set_xlabel("Sample")
            ax.set_ylabel(data_names[j])

        # Draw the plot.
        fig.canvas.draw()
        # Convert plot to image.
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # Resize the image.
        plot_img = cv2.resize(plot_img, image_lists[0][0].shape[:2])

        # Add the image to the list.
        data_plot_image_list.append(plot_img)

        # Clear the plot.
        plt.close(fig)

        # Create plot image for the trajectory.
        fig2, ax = plt.subplots()
        fig2.set_size_inches(5, 5)
        ax.plot(gt_traj_xyz[:, 0], gt_traj_xyz[:, 1], 'k', markersize=1)
        ax.plot(est_traj_xyz[:, 0], est_traj_xyz[:, 1], 'r', alpha=0.1)
        ax.plot(est_traj_xyz[:i, 0], est_traj_xyz[:i, 1], 'r', markersize=2)

        # Draw the plot.
        fig2.canvas.draw()
        # Convert plot to image.
        traj_img = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
        traj_img = traj_img.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
        # Resize the image.
        traj_img = cv2.resize(traj_img, image_lists[0][0].shape[:2])

        # Add the image to the list.
        traj_img_list.append(traj_img)

        # Clear the plot.
        plt.close(fig2)

    # Create the GIF.
    timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    gifname = "flow_and_losses" + timenow + ".gif"
    gif_path =  output_fpath.replace(".png", ".gif")

    # WHOOPS(yoraish): image labels currently unused.
    create_gif_from_image_lists(image_lists + [ traj_img_list, data_plot_image_list], ['raw', 'traj', 'est flow', 'stats'], gif_path, fps=10)

if __name__ == "__main__":
    # Create a dummy image list.
    est_traj_path = "/Users/pro/project/tartanair_tools/results_tartanair/Data_easy_P000_taext.txt"
    gt_traj_path = "/Users/pro/project/tartanair_tools/results_tartanair/Data_easy_P000_gt.txt"
    image_root_path = "/Users/pro/project/tartanair_tools/P000_image_lcam_front"

    images_path = glob.glob(os.path.join(image_root_path, "*.png"))
    images_path.sort()
    img_len = len(images_path)
    image_list = []

    T_NED_EDN = np.array([[0,1,0,0],
                        [0,0,1,0],
                        [1,0,0,0],
                        [0,0,0,1]], dtype=np.float32)
    T_NED_EDN = pp.mat2SE3(T_NED_EDN)
    
    for i, p in enumerate(images_path):
        im = cv2.imread(p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        image_list.append(im)

    # Create a dummy data list.
    data_list1 = np.zeros(img_len)
    data_list2 = np.zeros(img_len)

    # Create result.
    est_traj = np.loadtxt(est_traj_path)
    gt_traj = np.loadtxt(gt_traj_path)

    est_traj = pp.SE3(est_traj)
    gt_traj = pp.SE3(gt_traj)
    init = est_traj[0]

    est_traj = init.Inv() @ est_traj
    est_traj = est_traj.numpy()
    est_traj[:,:3] = est_traj[:,:3] * 2.366322147886725
    est_traj = pp.SE3(est_traj)
    est_traj = init @ est_traj

    # gt_traj = gt_traj[0].Inv() @ gt_traj

    est_traj = est_traj.numpy()
    gt_traj = gt_traj.numpy()

    results = {
        'est_traj': est_traj,
        'gt_traj': gt_traj
    }

    # Create the GIF.
    create_trajectory_summary_video(results, [image_list], [data_list1, data_list2], ['samples', 'samples'], "test.gif")
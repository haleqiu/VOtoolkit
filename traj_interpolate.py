import pypose as pp
import torch
import argparse
import glob,os

import numpy as np
import torch
import pypose as pp
fps = 1


T_NED_EDN = np.array([[0,1,0,0],
                    [0,0,1,0],
                    [1,0,0,0],
                    [0,0,0,1]], dtype=np.float32)
T_NED_EDN = pp.mat2SE3(T_NED_EDN)
 
def qinterp(qs, t, t_int):
    idxs = np.searchsorted(t, t_int)
    idxs0 = idxs-1
    idxs0[idxs0 < 0] = 0
    idxs1 = idxs
    idxs1[idxs1 == t.shape[0]] = t.shape[0] - 1
    q0 = qs[idxs0]
    q1 = qs[idxs1]
    tau = torch.zeros_like(t_int)
    dt = (t[idxs1]-t[idxs0])[idxs0 != idxs1]
    tau[idxs0 != idxs1] = (t_int-t[idxs0])[idxs0 != idxs1]/dt
    return slerp(q0, q1, tau)

def slerp(q0, q1, tau, DOT_THRESHOLD = 0.9995):
    """Spherical linear interpolation."""

    dot = (q0*q1).sum(dim=1)
    q1[dot < 0] = -q1[dot < 0]
    dot[dot < 0] = -dot[dot < 0]

    q = torch.zeros_like(q0)
    tmp = q0 + tau.unsqueeze(1) * (q1 - q0)
    tmp = tmp[dot > DOT_THRESHOLD]
    q[dot > DOT_THRESHOLD] = tmp / tmp.norm(dim=1, keepdim=True)

    theta_0 = dot.acos()
    sin_theta_0 = theta_0.sin()
    theta = theta_0 * tau
    sin_theta = theta.sin()
    s0 = (theta.cos() - dot * sin_theta / sin_theta_0).unsqueeze(1)
    s1 = (sin_theta / sin_theta_0).unsqueeze(1)
    q[dot < DOT_THRESHOLD] = ((s0 * q0) + (s1 * q1))[dot < DOT_THRESHOLD]
    return q / q.norm(dim=1, keepdim=True)

def xyz_interp(time, opt_time, xyz):
    '''
    Interpolate the xyz with the opt_time to the time
    '''
    intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
    intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
    intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])

    inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
    return torch.tensor(inte_xyz)

def interpolate_traj(traj, time_stamp, gt_time):
    '''
    interpolate the estimated trajectory to the gt trajectory
    this function will pad the begining and the end of the trajectory
    '''
    t_start = max(time_stamp[0], gt_time[0])
    t_end = min(time_stamp[-1], gt_time[-1])

    idx_start_traj = np.searchsorted(time_stamp, t_start)
    idx_end_traj = np.searchsorted(time_stamp, t_end, 'right')

    idx_start_gt = np.searchsorted(gt_time, t_start)
    idx_end_gt = np.searchsorted(gt_time, t_end, 'right')

    traj_ext = np.zeros((gt_time.shape[0], 7))

    # in case that the traj is outside of the gt
    traj = traj[idx_start_traj:idx_end_traj]
    print("idx_start_traj", idx_start_traj, "idx_end_traj", idx_end_traj)
    print("idx_start_gt", idx_start_gt, "idx_end_gt", idx_end_gt)

    rot_traj = traj.rotation()
    interp_rot = qinterp(rot_traj, 
                         torch.tensor(time_stamp[idx_start_traj:idx_end_traj]), 
                         torch.tensor(gt_time[idx_start_gt:idx_end_gt]))
    interp_rot = pp.SO3(interp_rot)
    interp_xyz = xyz_interp(gt_time[idx_start_gt:idx_end_gt], 
                            time_stamp[idx_start_traj:idx_end_traj], 
                            traj.translation())
    
    traj_ext[idx_start_gt:idx_end_gt] = np.hstack((interp_xyz, interp_rot.tensor()))
    # pad the the begining and the end
    traj_ext[:idx_start_gt] = traj_ext[idx_start_gt]
    traj_ext[idx_end_gt:] = traj_ext[idx_end_gt]

    return {'raw_traj': traj.tensor(), 'ext_traj': traj_ext, 'int_traj': traj_ext[idx_start_gt:idx_end_gt]}
    # np.savetxt(os.path.join(data_folder_path, path_name + "_taraw.txt"), traj_pp.tensor())
    # np.savetxt(os.path.join(data_folder_path, path_name + "_taext.txt"), dso_traj_ext)
    # np.savetxt(os.path.join(data_folder_path, path_name + "_gtcrop.txt"), gt_traj[time_stamp.astype(int)])
    # print("results_tartanair_seg/" + path_name + "_taext.txt")

if __name__ == "__main__":
    # Create a dummy image list.
    parser = argparse.ArgumentParser()
    parser.add_argument("--est_traj_path", type=str, default="/Users/pro/project/data/results_tartanair_seg300/Data_easy_P001_dataset_0_out.txt", help="path for estimated trajectory")
    parser.add_argument("--gt_traj_path", type=str, default="/Users/pro/project/data/results_tartanair_seg300/Data_easy_P001_dataset_0_gt.txt", help="gt path for ground truth trajectory")
    parser.add_argument("--exp_path", type=str, default="/Users/pro/project/data/results_tartanair_seg300", help="output file path")
    args = parser.parse_args(); print(args)

    # frames, key frames, time, qx, qy, qz, qw, x, y, z
    traj = np.loadtxt(args.est_traj_path)
    #  x, y, z, qx, qy, qz, qw
    gt_traj = np.loadtxt(args.gt_traj_path)

    traj_raw = np.zeros((traj.shape[0], 7))
    # extend the estimated trajectory to the ground truth trajectory
    gt_traj_ext = np.zeros((gt_traj.shape[0], 7))

    ## TODO: the time stamp of the gt and est is simulated
    time_stamp = traj[:,0].astype("float32")
    gt_time_stamp = np.arange(0, gt_traj_ext.shape[0], 1).astype("float32")
    print(time_stamp.dtype, gt_time_stamp.dtype)

    traj_raw[:, 0:3] = traj[:, 3:6]
    traj_raw[:, 3:] = traj[:, 6:10]
    traj_pp = pp.SE3(traj_raw).float()

    ## Transform to the gt frame for the
    traj_pp = (T_NED_EDN.Inv() @ traj_pp @ T_NED_EDN)
    gt_traj = pp.SE3(gt_traj).float()
    traj_pp = gt_traj[time_stamp[0].astype(int)] @ traj_pp
    
    ## the time stamp of the gt_traj
    out = interpolate_traj(traj_pp, time_stamp, gt_time_stamp)
    for k, v in out.items():
        print(k, v.shape)


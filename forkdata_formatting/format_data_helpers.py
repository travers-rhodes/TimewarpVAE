import numpy as np
import os

import forkdata_formatting.get_string_pixel_parameters as gspp

TRUNCATION_HEIGHT = 0.10
TRAJECTORY_LENGTH = 200


# This tip data is in vicon/world frame
# and is formatted as:
# secs,tx,ty,tz,rw,rx,ry,rz
def clean_trajectory_data(dat):
  # this choice is purely data-dependent,
  # but we can align our quats by forcing rz to start positive and then to 
  # choose sign of quat so that w moves as little as possible
  # data indices are
  # secs,tx,ty,tz,rw,rx,ry,rz
  rzind = 7
  for t in range(len(dat)):
   if t == 0:
       # ensure that w starts out positive 
       if dat[t, rzind] < 0:
           dat[t,4:] = -dat[t,4:]
       prevw = dat[t, rzind]
   else:
       if np.abs(-dat[t,rzind] - prevw) < np.abs(dat[t,rzind] - prevw):
           dat[t,4:] = -dat[t,4:]
       prevw = dat[t,rzind]

  tzind = 3 
  # remove data where z is above TRUNCATION HEIGHT
  # we're assuming that the data contains a single pass below TRUNCATION_HEIGHT
  first_good_t = np.where(dat[:,tzind] < TRUNCATION_HEIGHT)[0][0]
  first_bad_t = np.where(dat[(first_good_t):,tzind] > TRUNCATION_HEIGHT)[0][0] + first_good_t

  dat = dat[first_good_t:first_bad_t]

  # additionally, format the data into exactly
  xsamples, ysamples = lininterp(dat[:,0], dat[:,1:],TRAJECTORY_LENGTH)

  # ignore timing information for now and just return poses
  return ysamples

def load_trajectories_as_list(folder, ignorelist, max_traj_number=None):
  traj_index = 1
 
  all_string_info = []
  all_pose_data = []
  all_loaded_indices = []
  while True:
    image_file = folder + f"/overhead_image_{traj_index}.png"
    tip_poses_file = folder + f"/tip_pose_{traj_index}.npy"
    if os.path.exists(image_file) and os.path.exists(tip_poses_file) and traj_index not in ignorelist:
      plate, (mean_inds, axislens, th) = gspp.process_image(image_file)
      print(f"running for traj_index {traj_index}")
      # This tip data is in vicon/world frame
      # and is formatted as:
      # secs,tx,ty,tz,rw,rx,ry,rz
      tip_data = np.load(tip_poses_file) 
      # LINEAR TIME WARPING
      # format the data to a trajectory of fixed length with columns:
      # tx,ty,tz,rw,rx,ry,rz
      clean_pose_data = clean_trajectory_data(tip_data)
     
      all_string_info.append(np.array((mean_inds[0], mean_inds[1], axislens[0], axislens[1], th)))
      all_pose_data.append(clean_pose_data)
      all_loaded_indices.append(traj_index)
    else:
      if max_traj_number is None or traj_index > max_traj_number:
        break
    traj_index += 1

  all_string_info = np.array(all_string_info)
  all_pose_data = np.array(all_pose_data)
  all_loaded_indices = np.array(all_loaded_indices)

  return all_string_info, all_pose_data, all_loaded_indices


def format_all_trajectories_as_training_data(folder, savefilename):
  all_string_info, all_pose_data, all_loaded_indices = load_all_trajectories_as_lists(folder)

  # if we fail, idk, return None as an apology note?
  if all_pose_data.shape[0] <= 0:
    return None 

  # the mean of all x,y,z,rw,rx,ry,rz pointwise (over all trajectories)
  all_pose_data_mean = np.mean(all_pose_data, axis=(0,1)).reshape(1,7)
  all_pose_data_centered = all_pose_data - all_pose_data_mean
  ###
  ### For posterity, we note the typo in the next line, which means we scale
  ### to a somewhat arbitrary shape. it should have all_pose_data_centered[:,:,:3]
  ###
  position_std = np.sqrt(np.mean(np.var(all_pose_data_centered[:,:3],axis=0)))
  position_scale = 1./position_std
  # See the following jupyter notebook for an explanation of our rotation scaling parameter
  # http://localhost:8888/notebooks/data_processing/Rotation%20Scaling%20Scratchwork.ipynb
  rotation_scale = position_scale * 0.08
  all_pose_data_scaling = np.array((position_scale, position_scale, position_scale, rotation_scale,rotation_scale,rotation_scale, rotation_scale)).reshape(1,7)
  all_pose_data_scaled = all_pose_data_centered * all_pose_data_scaling

  np.savez(savefilename, 
          train_image_info = all_string_info, 
          train = all_pose_data_scaled,
          pose_scaling = all_pose_data_scaling,
          pose_mean = all_pose_data_mean)
  return savefilename

def lininterp(xvals, yvals, numsamples):
    newxvals = np.linspace(np.min(xvals), np.max(xvals), numsamples)
    newyvals = []
    curxind = 0
    for newxval in newxvals:
        while curxind < len(xvals) - 1 and xvals[curxind+1] < newxval:
            curxind += 1
        if curxind == len(xvals) - 1:
            newyval = yvals[curxind]
        else:
            ydiff = yvals[curxind + 1] - yvals[curxind]
            xdiff = xvals[curxind + 1] - xvals[curxind]
            if xdiff == 0:
                newyval = yvals[curxind]
            else:
                curxdiff = newxval - xvals[curxind]
                newyval = curxdiff/xdiff * ydiff + yvals[curxind]
                #print("curxdiff, xdiff, ydiff, newyval",curxdiff, xdiff, ydiff, newyval)
        newyvals.append(newyval)
    return newxvals, np.array(newyvals)

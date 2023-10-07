#!/usr/bin/env python3

import numpy as np
import pickle

# use the length of the wii remote as our length scale
savefilename = "data/trainTest2DLetterARescaled.npz"

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
        newyvals.append(newyval)
    return newxvals, np.array(newyvals)

with open("data/combined_trajectories.pickle","rb") as handle:
    data = pickle.load(handle)

rescaled_trajs = []
for i in range(len(data["gests"])):
    if data["chars"][i] != "A":
        continue
    traj = data["gests"][i].T
    _, rescaled_traj = lininterp(traj[:,0],traj[:,1:],200)
    rescaled_trajs.append(rescaled_traj)
rescaled_trajs = np.array(rescaled_trajs)

rng = np.random.default_rng(seed=42)
data_size = len(rescaled_trajs)
train_inds = rng.choice(data_size,int(data_size/2),replace=False)
test_inds = [i for i in range(data_size) if i not in train_inds]
# just 2D pos (ignore z and orientation and vels) 
train_pose_data = rescaled_trajs[train_inds,:,:2]
test = rescaled_trajs[test_inds,:,:2]


# the mean of all x,y,z,rw,rx,ry,rz pointwise (over all trajectories)
train_pose_data_mean = np.mean(train_pose_data, axis=(0,1)).reshape(1,1,2)
train_pose_data_centered = train_pose_data - train_pose_data_mean
position_std = np.sqrt(np.var(train_pose_data_centered[:,:,:2]))
position_scale = 1./position_std
train_pose_data_scaling = np.array((position_scale, position_scale)).reshape(1,1,2)
train_pose_data_scaled = train_pose_data_centered * train_pose_data_scaling

test_centered = test - train_pose_data_mean
test_scaled = test_centered * train_pose_data_scaling


np.savez(savefilename, 
      test = test_scaled,
      train = train_pose_data_scaled,
      pose_scaling = train_pose_data_scaling,
      pose_mean = train_pose_data_mean)

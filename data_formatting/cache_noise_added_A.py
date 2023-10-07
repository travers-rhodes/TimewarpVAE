#!/usr/bin/env python3

import numpy as np
import torch
import timewarp_lib.train_utils as tu

# use the length of the wii remote as our length scale
inputfilename = "data/trainTest2DLetterARescaled.npz"
savefilename = "data/trainTest2DLetterACache.npz"

np.random.seed(42)

data = np.load(inputfilename)
train_torch = torch.tensor(data["train"])
all_augmented_train = None
for _ in range(30):
  augmented_train = tu.add_timing_noise(train_torch,0.1).detach().cpu().numpy()
  if all_augmented_train is None:
    all_augmented_train = augmented_train
  else:
    all_augmented_train = np.concatenate((all_augmented_train, augmented_train),axis=0)


np.savez(savefilename, 
      test = data["test"],
      train = all_augmented_train,
      pose_scaling = data["pose_scaling"],
      pose_mean = data["pose_mean"])

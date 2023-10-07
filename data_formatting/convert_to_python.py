#!/usr/bin/env python3
# please run this file from the top-level directory
from scipy.io import loadmat
import numpy as np
import glob

for filename in glob.glob("data/matR_char/*.mat"):
  outfilename = filename[:-4]
  dat = loadmat(filename)
  np.savez_compressed(outfilename, **dat)


#!/usr/bin/env python3

import numpy as np
import pickle

all_gests  = []
all_biases = []
all_noises = []
filenames  = []
all_chars = []
all_participants = []
all_trials = []
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
  for case in ["upper"]:
    for participant in ["M1","T1","C1","Y1","Y2",
                        "J1","J2","J3","I1","C2",
                        "G1","C3","C4","G2","I2",
                        "A1","G3","U1","Y3","Z1",
                        "E1","S1","Z2","L1","I3"]:
      for trial in range(1,11):
        filename = f"{case}_{char}_{participant}_t{trial:02d}"
        datafile = f"data/matR_char_numpy/{filename}.npz"
        dat = np.load(datafile)
        all_gests.append(dat["gest"])
        all_biases.append(dat["bias"])
        all_noises.append(dat["noise"])
        all_chars.append(char)
        all_participants.append(participant)
        all_trials.append(trial)

with open("data/combined_trajectories.pickle","wb") as handle:
  pickle.dump({
      "gests": all_gests,
      "biases": all_biases,
      "noises": all_noises,
      "chars": all_chars,
      "trials": all_trials
    },
    handle)

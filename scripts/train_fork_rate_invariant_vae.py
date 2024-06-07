#!/usr/bin/env python3
import torch
import warnings

import timewarp_lib.training_template as att
import timewarp_lib.load_model as lm
from datetime import datetime
import numpy as np

import torch

SCRATCHFOLDER="../results/forkrateinvariantvae"
#os.makedirs(SCRATCHFOLDER)

warnings.filterwarnings("ignore","Initializing zero-element tensors is a no-op")

TRAJ_LEN=200
NUM_CHANNELS = 7
DATAFILE=f"../forkdata/forkTrajectoryData.npz"

NUM_EPOCHS = 10000

def train_and_save(latent_dim, paramdict, training_data_added_timing_noise):
  # https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
  timestr = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
  log_dir = f"{SCRATCHFOLDER}/{timestr}/log"

  MODELSAVEDIR=f"{SCRATCHFOLDER}/{timestr}/savedmodel"

  att.train_model(datafile = DATAFILE,
     model_save_dir = MODELSAVEDIR,
     num_epochs = NUM_EPOCHS,
     latent_dim = latent_dim,
     ## Generic
     device="cuda",
     dtype = torch.float,
     traj_len = TRAJ_LEN,
     traj_channels = NUM_CHANNELS,
     ## OptimizationRelated
     training_data_added_timing_noise = training_data_added_timing_noise,
     logname = log_dir,
     batch_size=64,
     log_to_wandb_name = "forkrateinvariantvae",
     **paramdict
     )

  _ = lm.LoadedModel(MODELSAVEDIR)

import base_configs as bc

for _ in range(5):
  for beta in [1.,0.1,0.01]:
    for latent_dim in [3]:
      for training_data_added_timing_noise in [0.1]:
        for paramdict in [bc.rate_invariant_autoencoder]:
          paramdict["beta"]=beta
          paramdict["emb_activate_last_layer"] = False
          paramdict["decoder_name"]="rate_invariant_conv"
          paramdict["use_rate_invariant_autoencoder"]=False
          paramdict["use_rate_invariant_vae"]=True
          paramdict["force_autoencoder"]=False
          paramdict["dec_gen_conv_layers_channels"] = [32,32,NUM_CHANNELS]
          train_and_save(latent_dim, paramdict, training_data_added_timing_noise)

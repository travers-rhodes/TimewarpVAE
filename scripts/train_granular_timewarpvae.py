#!/usr/bin/env python3
import torch
import warnings

import timewarp_lib.training_template as tt
import timewarp_lib.load_model as lm
from datetime import datetime
import numpy as np

import torch

SCRATCHFOLDER="../results/rateinvariantvae"
#os.makedirs(SCRATCHFOLDER)

warnings.filterwarnings("ignore","Initializing zero-element tensors is a no-op")

TRAJ_LEN=200
NUM_CHANNELS = 2
DATAFILE=f"../data/trainTest2DLetterARescaled.npz"

NUM_EPOCHS = 20000

def train_and_save(latent_dim, beta, paramdict, training_data_added_timing_noise):
  # https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
  timestr = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
  log_dir = f"{SCRATCHFOLDER}/{timestr}/log"

  MODELSAVEDIR=f"{SCRATCHFOLDER}/{timestr}/savedmodel"

  tt.train_model(datafile = DATAFILE,
     model_save_dir = MODELSAVEDIR,
     num_epochs = NUM_EPOCHS,
     latent_dim = latent_dim,
     ## Generic
     device="cuda",
     dtype = torch.float,
     traj_len = TRAJ_LEN,
     traj_channels = NUM_CHANNELS,
     ## VAE
     beta = beta,
     ## OptimizationRelated
     training_data_added_timing_noise = training_data_added_timing_noise,
     logname = log_dir,
     batch_size=64,
     log_to_wandb_name = "rateinvariantvae",
     **paramdict
     )

  _ = lm.LoadedModel(MODELSAVEDIR)

import base_configs as bc

for _ in range(5):
  for beta in [0.001,0.01]:#[0.001, 0.01, 0.1]:
      for latent_dim in [5,16,1,8,2,3,12,4,10,6,14]:
         for training_data_added_timing_noise in [0.1]:
           for dec_side_hiddens in [[200]]: 
             paramdict = bc.func_side_tw
             paramdict["dec_complicated_function_hidden_dims"] = dec_side_hiddens
             paramdict["dec_use_tanh"] = True
             paramdict["dec_use_elu"] = False 
             paramdict["scaltw_granularity"] = 199
             paramdict["scaltw_emb_conv_layers_channels"] = [32,32,32]
             paramdict["scaltw_emb_conv_layers_strides"] = [1,1,1]
             paramdict["scaltw_emb_conv_layers_kernel_sizes"] = [16,16,16]
             paramdict["scaltw_emb_nonlinearity"] = "Tanh"
             paramdict["scaltw_emb_conv_padding"] = "same"
             paramdict["scaltw_emb_activate_last_layer"] = True 
             paramdict["scalar_timewarper_timereg"] = 0.005
             paramdict["emb_conv_layers_channels"] = [32,32,32]
             paramdict["emb_conv_layers_strides"] = [1,1,1]
             paramdict["emb_conv_layers_kernel_sizes"] = [16,16,16]
             paramdict["emb_conv1d_padding"] = "same"
             paramdict["emb_activate_last_layer"] = False
             paramdict["emb_nonlinearity"] = "Tanh"
             paramdict["force_autoencoder"] = False
             train_and_save(latent_dim, beta, paramdict, training_data_added_timing_noise)

#!/usr/bin/env python3
import torch
import warnings

import timewarp_lib.training_template as tt
import timewarp_lib.load_model as lm
from datetime import datetime
import numpy as np

import torch

SCRATCHFOLDER="results/rescaled"
#os.makedirs(SCRATCHFOLDER)

warnings.filterwarnings("ignore","Initializing zero-element tensors is a no-op")

TRAJ_LEN=200
NUM_CHANNELS = 2
DATAFILE=f"data/trainTest2DLetterARescaled.npz"

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
     log_to_wandb_name = "rescaled",
     **paramdict
     )

  _ = lm.LoadedModel(MODELSAVEDIR)

import base_configs as bc

for _ in range(5):
  for use_dtw in [True]:
    for beta in [0.001]:#, 0.01, 0.1, 0.0005]:
      for latent_dim in [5,16,1,8,2,3,12,4,10,6,14]:
        for training_data_added_timing_noise in [0.1]:#,0]:
           if beta != 0.001 and training_data_added_timing_noise == 0:
             continue
           for paramdict in [bc.func_side_no_tw]:
             for dec_side_hiddens in [[200]]: 
               if beta != 0.001 and len(dec_side_hiddens)==0:
                 continue
               if dec_side_hiddens != [200] and paramdict["decoder_name"] != "functional_decoder_complicated":
                 continue
               if paramdict["decoder_name"] == "functional_decoder_complicated":
                 paramdict["dec_complicated_function_hidden_dims"] = dec_side_hiddens
               if use_dtw:
                 if dec_side_hiddens != [200]:
                   continue
                 if beta != 0.001:
                   continue
                 ##OVERWRITE and do DTW for loss
                 paramdict["vector_timewarper_name"]="dtw_vector_timewarper"
                 paramdict["vector_timewarper_warps_recon_and_actual"]=True
               train_and_save(latent_dim, beta, paramdict, training_data_added_timing_noise)

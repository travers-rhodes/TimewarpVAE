import timewarp_lib.training_template as tt
import timewarp_lib.load_model as lm
import shutil
import os
import time

import numpy as np
import torch

## create a dataset doing something like
SCRATCHFOLDER="test/data/tmp"
DATAFILE=f"{SCRATCHFOLDER}/verysimpledata.npz"
MODELSAVEDIR=f"{SCRATCHFOLDER}/verysimplemodel"
TRAJ_LEN = 100
NUM_CHANNELS = 1


def test_new_architectures():
  for num_trajs in [1,4]:
    train_and_save(num_trajs)

def train_and_save(num_trajs):
  shutil.rmtree(SCRATCHFOLDER, ignore_errors=True)
  os.makedirs(SCRATCHFOLDER)
  data = np.array([np.abs(np.linspace(-1,1,TRAJ_LEN))**k for k in np.linspace(2,3,num_trajs)]).reshape(num_trajs,TRAJ_LEN,1).repeat(NUM_CHANNELS,axis=2)

  # the mean of all x,y,z,rw,rx,ry,rz pointwise (over all trajectories and times
  data_mean = np.mean(data, axis=(0,1)).reshape(1,1,NUM_CHANNELS)

  data_centered = data - data_mean
  std = np.sqrt(np.mean(np.var(data_centered,axis=0)))
  data_scaling = 1./std if std > 0 else 1
  data_scaled = data_centered * data_scaling
  np.savez(DATAFILE, 
         train=data_scaled,
         pose_scaling = data_scaling,
         pose_mean = data_mean)

  # https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
  timestr = time.strftime("%Y%m%d-%H%M%S")
  log_dir = f"{SCRATCHFOLDER}/log/{timestr}"

  tt.train_model(datafile = DATAFILE,
     model_save_dir = MODELSAVEDIR,
     num_epochs = 100,
     latent_dim = 0,
     ## Generic
     device="cpu",
     dtype = torch.float,
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###ModeledParameterScalarTimewarper###
     scalar_timewarper_name = "modeled_scalar_timewarper",
     traj_len = TRAJ_LEN,
     traj_channels = NUM_CHANNELS,
     scaltw_granularity = 20,
     scaltw_emb_conv_layers_channels = [10,10],
     scaltw_emb_conv_layers_strides = [2,2],
     scaltw_emb_conv_layers_kernel_sizes = [3,3],
     scaltw_emb_fc_layers_num_features = [32],
     #dtype=torch.float,
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     encoder_name="convolutional_encoder",
     #latent_dim,
     #traj_len,
     #traj_channels,
     emb_conv_layers_channels = [16,16],
     emb_conv_layers_strides = [2,2],
     emb_conv_layers_kernel_sizes = [3,3],
     emb_fc_layers_num_features = [16],
     #dtype=torch.float,
     ###########################
     #########Decoding##########
     ###########################
     ###OneDConvDecoder###
     decoder_name="convolutional_decoder",
     #latent_dim,
     dec_gen_fc_layers_num_features = [int(TRAJ_LEN/4) * 16],
     dec_gen_first_traj_len=int(TRAJ_LEN/4),
     dec_gen_conv_layers_channels = [32, 32, NUM_CHANNELS],
     # first_traj_len times all strides should give final size
     dec_gen_conv_layers_strides = [2,2,2],
     dec_gen_conv_layers_kernel_sizes = [5,5,5],
     #dtype=torch.float,
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityTimewarper### 
     vector_timewarper_name="identity_vector_timewarper",
     ## VAE
     beta = 0.01,
     ## TimeWarpingRelated
     scalar_timewarper_timereg = 0,
     ## OptimizationRelated
     # Set time_optim_lr to zero if you don't want to learn it
     pre_time_learning_epochs = 0,
     scalar_timewarping_lr = 0.01,
     scalar_timewarping_eps = 0.000001,
     decoding_lr = 0.001,
     encoding_lr = 0.001,
     decoding_eps = 0.000001,
     encoding_eps = 0.000001,
     useAdam = True,
     use_timewarp_loss = True,
     logname = log_dir,
     batch_size=64,
     curv_loss_penalty_weight=0,
     )

  loaded_model = lm.LoadedModel(MODELSAVEDIR)

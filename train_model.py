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

def train_and_save(paramlr,latent_dim, beta, paramdict, curv_loss_lambda, div_by_zero_epsilon, time_scale_lr, time_regularization, time_endpoint_regularization, use_softplus, use_elu,
                   scaltw_min_canonical_time, scaltw_max_canonical_time, decoding_l2_weight_decay,
                   dec_spatial_regularization_factor, decoding_spatial_derivative_regularization,
                   emb_dropout,
                   training_data_added_timing_noise
                   ):
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
     pre_time_learning_epochs = 0,
     scalar_timewarping_lr = time_scale_lr,
     scalar_timewarping_eps = 0.000001,
     scalar_timewarper_timereg = time_regularization,
     scalar_timewarper_endpointreg = time_endpoint_regularization,
     scaltw_min_canonical_time = scaltw_min_canonical_time,
     scaltw_max_canonical_time = scaltw_max_canonical_time,
     dec_use_softplus=use_softplus,
     dec_use_elu=use_elu,
     dec_conv_use_elu=use_elu,
     decoding_l2_weight_decay=decoding_l2_weight_decay,
     decoding_spatial_derivative_regularization=decoding_spatial_derivative_regularization,
     dec_spatial_regularization_factor=dec_spatial_regularization_factor,
     decoding_lr = paramlr,
     encoding_lr = paramlr,
     decoding_eps = 0.0001,
     encoding_eps = 0.0001,
     useAdam = True,
     logname = log_dir,
     batch_size=64,
     curv_loss_penalty_weight = curv_loss_lambda,
     curv_loss_epsilon_scale = 0.01,
     curv_loss_num_new_sampling_points = 1000,
     curv_loss_divide_by_zero_epsilon = div_by_zero_epsilon,
     log_to_wandb_name = "rescaled",
     emb_dropout_probability = emb_dropout,
     **paramdict
     )

  _ = lm.LoadedModel(MODELSAVEDIR)

import base_configs as bc

curv_loss_penalty_weight =0#0.000000001,0.00000001,0]:
div_by_zero_epsilon  = 1e-7
lr = 0.0001
time_endpoint_regularization = 0
# 1.0 here means do nothing (multiplicative factor, you silly goose)
dec_spatial_regularization_factor = 1.0#3.0, 10.0, 2.0]:
for _ in range(5):
    for beta in [0.001, 0.01, 0.1]:
      for latent_dim in [5,16,1,8,2,3,12,4,10,6,14]:
         for training_data_added_timing_noise in [0.1,0]:
           if beta != 0.001 and training_data_added_timing_noise == 0:
             continue
           desired_time_regularization = 0.05
           desired_time_scale_lr = 0.0001
           scaltw_min_canonical_time = 0.0
           scaltw_max_canonical_time = 1.0
           decoding_spatial_derivative_regularization = 0.0
           decoding_l2_weight_decay = 0.0
           emb_dropout  = 0
           for paramdict in [bc.func_side_tw]:#, bc.func_side_no_tw]: 
             for dec_side_hiddens in [[200],[]]: 
               if beta != 0.001 and len(dec_side_hiddens)==0:
                 continue
               paramdict["dec_complicated_function_hidden_dims"] = dec_side_hiddens
               use_softplus = False
               use_elu = True
               time_scale_lr = desired_time_scale_lr
               time_regularization = desired_time_regularization
               paramdict["step_each_batch"] = True
               paramdict["learn_decoder_variance"] = False 
               paramdict["dec_initial_log_noise_estimate"] = np.log(0.1**2).item()
               train_and_save(lr, latent_dim, beta, paramdict,curv_loss_penalty_weight,div_by_zero_epsilon, time_scale_lr, time_regularization, time_endpoint_regularization, use_softplus, use_elu, scaltw_min_canonical_time,scaltw_max_canonical_time, decoding_l2_weight_decay, dec_spatial_regularization_factor, decoding_spatial_derivative_regularization, emb_dropout, training_data_added_timing_noise)

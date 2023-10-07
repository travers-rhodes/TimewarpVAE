#!/usr/bin/env python3
import torch
import warnings

import timewarp_lib.training_template as tt
import timewarp_lib.load_model as lm
from datetime import datetime
import numpy as np

import torch

SCRATCHFOLDER="results/overnight"
#os.makedirs(SCRATCHFOLDER)

warnings.filterwarnings("ignore","Initializing zero-element tensors is a no-op")

TRAJ_LEN=200
NUM_CHANNELS = 2
DATAFILE=f"data/trainTest2DLetterAScaled.npz"

NUM_EPOCHS = 20000

def train_and_save(paramlr,latent_dim, beta, paramdict, curv_loss_lambda, div_by_zero_epsilon, time_scale_lr, time_regularization, time_endpoint_regularization, use_softplus, use_elu,
                   scaltw_min_canonical_time, scaltw_max_canonical_time, decoding_l2_weight_decay,
                   dec_spatial_regularization_factor, decoding_spatial_derivative_regularization,
                   dec_template_use_custom_initialization,
                   dec_template_custom_initialization_grad_t,
                   dec_template_custom_initialization_t_intercept_padding,
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
     dec_template_use_custom_initialization = dec_template_use_custom_initialization,
     dec_template_custom_initialization_grad_t = dec_template_custom_initialization_grad_t,
     dec_template_custom_initialization_t_intercept_padding = dec_template_custom_initialization_t_intercept_padding,
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
     log_to_wandb_name = "overnight",
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
  for emb_nonlinearity in ["ReLU"]:
    for beta in [0.001, 0.01, 0.1]:
      for latent_dim in [5,16,1,8,2,3,12,4,10,6,14]:
         for training_data_added_timing_noise in [0.1]:
           if beta != 0.001 and training_data_added_timing_noise == 0:
             continue
           for desired_time_regularization in [0.05]:
              for desired_time_scale_lr in [0.0001]:
                for desired_dec_template_custom_initialization_grad_t in [10.0]:
                  desired_dec_template_use_custom_initialization = False
                  if desired_dec_template_custom_initialization_grad_t is not None:
                    desired_dec_template_use_custom_initialization = True
                  dec_template_custom_initialization_t_intercept_padding = 0.1
                  scaltw_min_canonical_time = 0.0
                  scaltw_max_canonical_time = 1.0
                  for decoding_spatial_derivative_regularization in [0.0]:
                    for decoding_l2_weight_decay in [0.0]:
                      if ((scaltw_min_canonical_time != -scaltw_max_canonical_time 
                          and scaltw_min_canonical_time != 0
                          and scaltw_max_canonical_time != 0)
                         or scaltw_max_canonical_time == scaltw_min_canonical_time):
                        continue
                      for emb_dropout in [0]:
                        for paramdict in [bc.func_side_no_tw, bc.convup_no_dtw]: 
                            paramdict["dec_complicated_only_side_latent"] = True
                            for dec_side_hiddens in [[200],[]]: 
                              if beta != 0.001 and len(dec_side_hiddens)==0:
                                continue
                              if beta != 0.001 and len(dec_side_hiddens)==0:
                                continue
                              dec_template_hiddens = [500,500]
                              if desired_time_regularization != 0.05 and paramdict["scalar_timewarper_name"] == "identity_scalar_timewarper":
                                # don't multiple-do no timewarping
                                continue
                              for emb_conv_fc_size in [0]:
                                paramdict["emb_nonlinearity"] = emb_nonlinearity

                                if paramdict["decoder_name"] =="functional_decoder" or paramdict["decoder_name"] == "functional_decoder_complicated":
                                  paramdict["dec_template_motion_hidden_layers"]=dec_template_hiddens
                                  dec_template_custom_initialization_grad_t = desired_dec_template_custom_initialization_grad_t
                                  dec_template_use_custom_initialization = desired_dec_template_use_custom_initialization
                                elif dec_template_hiddens != [500,500]:
                                  continue
                                elif dec_side_hiddens != [200]:
                                  continue
                                elif desired_dec_template_custom_initialization_grad_t != 10.0:
                                  continue
                                else:
                                  # if not functional decoder, then no template
                                  dec_template_custom_initialization_grad_t = 0
                                  dec_template_use_custom_initialization = False
                                if paramdict["decoder_name"] == "convolutional_decoder":
                                  NUM_CHANNELS = 2
                                  paramdict["dec_gen_fc_layers_num_features"] = [30*NUM_CHANNELS*10]
                                  paramdict["dec_gen_first_traj_len"]=30
                                  paramdict["dec_gen_conv_layers_channels"] = [8,8,NUM_CHANNELS]
                                  paramdict["dec_gen_conv_layers_strides"] = [2,2,2]
                                  paramdict["dec_gen_conv_layers_kernel_sizes"] = [3,3,3]


                                if paramdict["decoder_name"] == "functional_decoder_complicated":
                                  paramdict["dec_complicated_function_hidden_dims"] = dec_side_hiddens
                                  paramdict["dec_complicated_function_latent_size"] = 64
                                if paramdict["scalar_timewarper_name"] == "modeled_scalar_timewarper":
                                  paramdict["scaltw_granularity"] = 50
                                  paramdict["scaltw_emb_conv_layers_channels"] = [16,32,32,64,64,64]
                                  paramdict["scaltw_emb_conv_layers_strides"] = [1,2,1,2,1,2]
                                  paramdict["scaltw_emb_conv_layers_kernel_sizes"] = [3,3,3,3,3,3]
                                  paramdict["scaltw_emb_fc_layers_num_features"] = []
                                if paramdict["encoder_name"] == "convolutional_encoder":
                                  paramdict["emb_conv_layers_channels"] = [16,32,64,32]
                                  paramdict["emb_conv_layers_strides"] = [1,2,2,2] 
                                  paramdict["emb_conv_layers_kernel_sizes"] = [3,3,3,3]
                                if emb_conv_fc_size != 0:
                                  paramdict["emb_fc_layers_num_features"] = [emb_conv_fc_size]
                                else:
                                  paramdict["emb_fc_layers_num_features"] = []
                                for use_softplus in [False]:
                                  for use_elu in [True]:
                                    if use_elu and use_softplus:
                                      continue
                                    if paramdict["scalar_timewarper_name"] == "identity_scalar_timewarper":
                                      time_scale_lr = 0
                                      time_regularization = 0.0
                                    else:
                                      time_scale_lr = desired_time_scale_lr
                                      time_regularization = desired_time_regularization
                                    paramdict["step_each_batch"] = True
                                    paramdict["learn_decoder_variance"] = False 
                                    #paramdict["noise_lr"] = 0.00001
                                    #paramdict["noise_eps"] = 0.0001
                                    paramdict["dec_initial_log_noise_estimate"] = np.log(0.1**2).item()
                                    train_and_save(lr, latent_dim, beta, paramdict,curv_loss_penalty_weight,div_by_zero_epsilon, time_scale_lr, time_regularization, time_endpoint_regularization, use_softplus, use_elu, scaltw_min_canonical_time,scaltw_max_canonical_time, decoding_l2_weight_decay, dec_spatial_regularization_factor, decoding_spatial_derivative_regularization, dec_template_use_custom_initialization, dec_template_custom_initialization_grad_t,dec_template_custom_initialization_t_intercept_padding, emb_dropout, training_data_added_timing_noise)

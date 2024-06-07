import torch
import unittest
import timewarp_lib.autoencoder_training_template as att
import timewarp_lib.load_model as lm
import time
import shutil
import os
import numpy as np

## create a dataset doing something like
SCRATCHFOLDER="test/data/tmp"
DATAFILE=f"{SCRATCHFOLDER}/verysimpledata.npz"
MODELSAVEDIR=f"{SCRATCHFOLDER}/verysimplemodel"
TRAJ_LEN = 100
NUM_CHANNELS = 1


NUM_EPOCHS = 10

class TestRIALearner(unittest.TestCase):
  def test_ria_integration(self):
    num_trajs = 4

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
         test=data_scaled,
         pose_scaling = data_scaling,
         pose_mean = data_mean)
    # https://stackoverflow.com/questions/10607688/how-to-create-a-file-name-with-the-current-date-time-in-python
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"{SCRATCHFOLDER}/log/{timestr}"
    args = dict(
     use_rate_invariant_autoencoder = True,
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
     learn_decoder_variance=False,
     training_data_added_timing_noise=0.0,
     noise_lr=0.0,
     noise_eps=0.0,
     decoding_l2_weight_decay=0.0,
     decoding_spatial_derivative_regularization=0.0,
     step_each_batch=True,
     datafile = DATAFILE,
     model_save_dir = MODELSAVEDIR,
     num_epochs = NUM_EPOCHS,
     ## Generic
     device="cpu",
     dtype = torch.float,
     traj_len = TRAJ_LEN,
     traj_channels = NUM_CHANNELS,
     ## VAE
     ## OptimizationRelated
     latent_dim = 10,
     ria_T = 99,
     ###########################
     #####ScalarTimeWarping#####
     ###########################
     ###IdentityScalarTimewarper###
     scalar_timewarper_name = "identity_scalar_timewarper",
     ###########################
     #########Encoding##########
     ###########################
     ###OneDConvEncoder###
     encoder_name="convolutional_encoder",
     emb_conv_layers_channels = [32,32,32],#[],#
     emb_conv_layers_strides = [1,1,1],#[],#
     emb_conv_layers_kernel_sizes = [16,16,16],#[],#
     emb_fc_layers_num_features = [],#[],#
     emb_nonlinearity = "Tanh",
     emb_activate_last_layer = True,
     emb_conv1d_padding = "same",
     ###########################
     #########Decoding##########
     ###########################
     ###OneDConvDecoder###
     decoder_name="convolutional_decoder_upsampling",
     dec_gen_fc_layers_num_features = [TRAJ_LEN*32],
     dec_gen_first_traj_len=TRAJ_LEN,
     dec_gen_conv_layers_channels = [32,32,NUM_CHANNELS],
     dec_gen_upsampling_factors = [1,1,1],
     dec_gen_conv_layers_kernel_sizes = [16,16,16],
     dec_use_tanh = True,
     dec_conv1d_padding = "same",
     ###########################
     #####VectorTimewarping#####
     ###########################
     ###IdentityVectorTimewarper### 
     vector_timewarper_name="identity_vector_timewarper",
     )
    att.train_model(**args)

    _ = lm.LoadedModel(saved_model_dir=MODELSAVEDIR)
     

if __name__ == '__main__':
  unittest.main()



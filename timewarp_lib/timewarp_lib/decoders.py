import torch
import torch.nn as nn
import math
import numpy as np

import timewarp_lib.utils.function_style_template_motion as tm
import timewarp_lib.parameterized_vector_time_warper as pvtw
# This learns a function that takes in as input a time and returns the pose at that time.
# This is different from a ConvStyleDecoder which returns several poses at fixed timesteps.
class ComplicatedFunctionStyleDecoder(nn.Module):
    def __init__(self,
        latent_dim,
        traj_channels,
        dtype=torch.float,
        dec_template_motion_hidden_layers=[1000],
        dec_use_softplus = False,
        dec_use_elu = False,
        dec_use_tanh = False,
        dec_template_use_custom_initialization = False,
        # grad_t is the constant, relatively large, abs value of default slope of first 
        # linear layer in the $t$ direction
        # t_intercept_padding is padding such that the first linear layer
        # has t-intercept uniformly distributed in the interval [-t_intercept_padding, 1+t_intercept_padding]
        dec_template_custom_initialization_grad_t = None,
        dec_template_custom_initialization_t_intercept_padding = None,
        dec_complicated_function_hidden_dims = [16],
        dec_complicated_function_latent_size = 16,
        dec_complicated_only_side_latent = False,
        dec_initial_log_noise_estimate = None,
        **kwargs):
       super(ComplicatedFunctionStyleDecoder, self).__init__()

       # as in 
       # "Connections with Robust PCA and the Role of Emergent
       # Sparsity in Variational Autoencoder Models"
       # specifically parameterize the decoder gaussian variance
       #. INITIALIZATION note: initialize small (roughly equal to expected noise of data)
       # also, for backward compatibility, only include this as a Parameter (required for loading from saved file)
       # if it was initialized when created
       if dec_initial_log_noise_estimate is not None:
         dec_initial_log_noise_estimate = np.array(dec_initial_log_noise_estimate).item() # round-trip idempotently to just a number
         self.log_decoder_variance = nn.parameter.Parameter(data=torch.FloatTensor((dec_initial_log_noise_estimate,)), requires_grad=True)
       else:
         self.log_decoder_variance = torch.FloatTensor((1.,))

       template_motion_hidden_layers = list(dec_template_motion_hidden_layers)

       template_input_dim = 0 if dec_complicated_only_side_latent else latent_dim
       self.motion_model = tm.TemplateMotionGeneration(input_dim=template_input_dim,
                  passthrough_dim=1,
                  layer_widths= template_motion_hidden_layers + [dec_complicated_function_latent_size],
                  use_softplus = dec_use_softplus,
                  use_elu = dec_use_elu,
                  use_tanh = dec_use_tanh,
                  use_custom_initialization=dec_template_use_custom_initialization,
                  custom_initialization_grad_t=dec_template_custom_initialization_grad_t,
                  custom_initialization_t_intercept_padding=dec_template_custom_initialization_t_intercept_padding,
                  dtype=dtype)
  
       self.nonlinearity = torch.nn.Softplus() if dec_use_softplus else (
           torch.nn.ELU() if dec_use_elu else (
             (torch.nn.Tanh() if dec_use_tanh else torch.nn.ReLU())))
        
       previous_layer_width = latent_dim
       self.all_side_layers = []
       for w in dec_complicated_function_hidden_dims:
          self.all_side_layers.append(nn.Linear(previous_layer_width, w))
          previous_layer_width = w
       self.all_side_layers.append(nn.Linear(previous_layer_width, 
                                             dec_complicated_function_latent_size * traj_channels))
       self.all_side_layers = nn.ModuleList(self.all_side_layers)

       self.dec_complicated_function_latent_size = dec_complicated_function_latent_size 
       self.traj_channels = traj_channels
       self.only_side_latent = dec_complicated_only_side_latent 

    def get_mean_square_layer_weights(self):
      return self.motion_model.mean_square_layer_weights()

    # given a batch of zs of shape (batchsize, latent_dim)
    def decode_and_return_noisy_embedding_node(self, zs, ts):
      assert len(ts.shape)==3  and ts.shape[2] == 1, "we're assuming you're passing in a single set of ts with shape (batchsize, timevalues, timechannel=1)"
      assert zs.shape[0] == ts.shape[0], "we're assuming you're passing in zs and ts of the same shape"

      batchsize = ts.shape[0]
      timesteps = ts.shape[1]
      latent_dim = zs.shape[1]
      broadcast_zs = zs.reshape((batchsize,1,latent_dim)).expand((batchsize,timesteps,latent_dim))
      broadcast_zs = broadcast_zs.reshape((batchsize*timesteps,latent_dim))
      ts = ts.reshape((batchsize*timesteps,1))
      # The shape of trajectory is now 
      #          (batchsize*timesteps,
      #           dec_complicated_function_latent_size))
      added_motion_info = torch.empty(size=(batchsize*timesteps,0),dtype=zs.dtype, device=zs.device) if self.only_side_latent else broadcast_zs 
      trajectory = self.motion_model(added_motion_info, ts)


      # Now, compute the side-channel results
      layer = broadcast_zs
      for i, fc in enumerate(self.all_side_layers):
          if i != len(self.all_side_layers) - 1:
              layer = self.nonlinearity(fc(layer))
          else: # don't nonlinear the last layer
              layer = fc(layer)

      side_layer = layer.reshape((batchsize*timesteps, self.traj_channels, self.dec_complicated_function_latent_size))

      # Matrix multiply each of the batchsize*timesteps rows
      combined_traj = torch.einsum("bcd,bd->bc", side_layer, trajectory)

      # reshape back to (batchsize, timesteps, numoutchannels)
      return combined_traj.reshape((batchsize,timesteps,-1)), broadcast_zs
    
    def decode(self, zs, ts):
      trajectory, _ = self.decode_and_return_noisy_embedding_node(zs,ts)
      return trajectory

# This learns a function that takes in as input a time and returns the pose at that time.
# This is different from a ConvStyleDecoder which returns several poses at fixed timesteps.
class FunctionStyleDecoder(nn.Module):
    def __init__(self,
        latent_dim,
        traj_channels,
        dtype=torch.float,
        dec_template_motion_hidden_layers=[1000],
        dec_use_softplus = False,
        dec_use_elu = False,
        dec_spatial_regularization_factor = 1,
        dec_template_use_custom_initialization = False,
        # grad_t is the constant, relatively large, abs value of default slope of first 
        # linear layer in the $t$ direction
        # t_intercept_padding is padding such that the first linear layer
        # has t-intercept uniformly distributed in the interval [-t_intercept_padding, 1+t_intercept_padding]
        dec_template_custom_initialization_grad_t = None,
        dec_template_custom_initialization_t_intercept_padding = None,
        **kwargs):
       super(FunctionStyleDecoder, self).__init__()

       template_motion_hidden_layers = list(dec_template_motion_hidden_layers)

       self.motion_model = tm.TemplateMotionGeneration(input_dim=latent_dim,
                  passthrough_dim=1,
                  layer_widths= template_motion_hidden_layers + [traj_channels],
                  use_softplus = dec_use_softplus,
                  use_elu = dec_use_elu,
                  use_custom_initialization=dec_template_use_custom_initialization,
                  custom_initialization_grad_t=dec_template_custom_initialization_grad_t,
                  custom_initialization_t_intercept_padding=dec_template_custom_initialization_t_intercept_padding,
                  dtype=dtype)
  
       # an idempotent operation to convert the (possibly) numpy scalar or scalar to a scalar
       self.dec_spatial_regularization_factor = np.array(dec_spatial_regularization_factor).item()
       self.latent_dim = np.array(latent_dim).item()

    def get_mean_square_layer_weights(self):
      return self.motion_model.mean_square_layer_weights()

    # given a batch of zs of shape (batchsize, latent_dim)
    def decode_and_return_noisy_embedding_node(self, in_zs, ts):
      assert len(ts.shape)==3  and ts.shape[2] == 1, "we're assuming you're passing in a single set of ts with shape (batchsize, timevalues, timechannel=1)"
      assert in_zs.shape[0] == ts.shape[0], "we're assuming you're passing in zs and ts of the same shape"

      batchsize = ts.shape[0]
      timesteps = ts.shape[1]
      latent_dim = self.latent_dim
      # here, we add logic which is trivial/do nothing if
      # we're taking in a latent dim of the expected size.
      # However, if we're taking in a rate_invariant latent dim,
      # then we just want to take the last latent_dim values to decode
      # (this is only used if we're training a conv decoder on a rate_invariant
      # model). Not sure why we would ever do that.... but shh.
      layer = in_zs[:,-latent_dim:]
      broadcast_zs = layer.reshape((batchsize,1,latent_dim)).expand((batchsize,timesteps,latent_dim))
      broadcast_zs = broadcast_zs.reshape((batchsize*timesteps,latent_dim))
      # Dividing by the spatial regularization factor has the effect of
      # (by default, in order for training to have the model return the same pattern) 
      # multiplying the relevant coefficients by that factor, thus increasing any associated 
      # regularization quantity by this factor (squared, if L2 regularization)
      scaled_broadcast_zs = broadcast_zs / self.dec_spatial_regularization_factor
      ts = ts.reshape((batchsize*timesteps,1))
      # The shape of trajectory is now (batchsize*timesteps, numoutchannels)
      trajectory = self.motion_model(scaled_broadcast_zs, ts)
      # reshape back to (batchsize, timesteps, numoutchannels)
      return trajectory.reshape((batchsize,timesteps,-1)), broadcast_zs
    
    def decode(self, zs, ts):
      trajectory, _ = self.decode_and_return_noisy_embedding_node(zs,ts)
      return trajectory

# This is a standard 1D convolutional decoder
class OneDConvDecoder(nn.Module):
    def __init__(self,
            latent_dim, 
            dec_gen_fc_layers_num_features = [16*32],
            dec_gen_first_traj_len=16,
            dec_gen_conv_layers_channels = [32, 32, 7],
            # first_traj_len times all strides should give final size
            dec_gen_conv_layers_strides = [2,2,1],
            dec_gen_conv_layers_kernel_sizes = [5,5,5],
            dec_use_softplus = False,
            # shame on me---forgot to check this
            dec_conv_use_elu = False,
            dec_initial_log_noise_estimate = None,
            dtype=torch.float,
            **kwargs):
        super(OneDConvDecoder, self).__init__()
        if dec_initial_log_noise_estimate is not None:
          dec_initial_log_noise_estimate = np.array(dec_initial_log_noise_estimate).item() # round-trip idempotently to just a number
          self.log_decoder_variance = nn.parameter.Parameter(data=torch.FloatTensor((dec_initial_log_noise_estimate,)), requires_grad=True)
        else:
          self.log_decoder_variance = torch.FloatTensor((1.,))
        self.latent_dim = latent_dim
        self.gen_fc_layers_num_features = dec_gen_fc_layers_num_features
        self.gen_first_traj_len= dec_gen_first_traj_len
        self.gen_conv_layers_channels = dec_gen_conv_layers_channels
        self.gen_conv_layers_strides = dec_gen_conv_layers_strides
        self.gen_conv_layers_kernel_sizes = dec_gen_conv_layers_kernel_sizes 
        # construct the parameters for all the generative fully-connected layers
        self.gen_fcs = []
        prev_features = self.latent_dim
        layer_features = prev_features
        for layer_features in self.gen_fc_layers_num_features:
            self.gen_fcs.append(nn.Linear(prev_features, layer_features,dtype=dtype))
            prev_features = layer_features
        # construct the parameters for all the generative convolutions
        traj_len = self.gen_first_traj_len
        self.gen_first_conv_channels = int(layer_features/traj_len )
        prev_channels = self.gen_first_conv_channels
        self.gen_convs = []
        self.gen_conv_crops = []
        # 2022_11_23: Investigation discovered that zero padding makes no sense for the edges of trajectories
        # so we want to be careful here and truncate off the edges which are using more zero-padding than the internal
        # datapoints are.
        # I've decided that safe-padding means we should only include the middle
        # desired_traj_len = (input_len+1) * stride - kernel_size points.
        for i, layer_channels in enumerate(self.gen_conv_layers_channels):
            pytorch_traj_len = (traj_len - 1) * self.gen_conv_layers_strides[i] + (self.gen_conv_layers_kernel_sizes[i] - 1) + 1
            # based on exploration of edge points which have ``more'' zero-padding than comparable internal points
            # note that this removes exactly:
            #(traj_len - 1) * self.gen_conv_layers_strides[i] + (self.gen_conv_layers_kernel_sizes[i] - 1) + 1
            #-(traj_len + 1) * self.gen_conv_layers_strides[i] - self.gen_conv_layers_kernel_sizes[i]
            # = 
            # 2 * (self.gen_conv_layers_kernel_sizes[i] - self.gen_conv_layers_strides[i])
            # points
            desired_traj_len = (traj_len + 1) * self.gen_conv_layers_strides[i] - self.gen_conv_layers_kernel_sizes[i]
            self.gen_convs.append(
                    nn.ConvTranspose1d(prev_channels,
                              layer_channels, 
                              self.gen_conv_layers_kernel_sizes[i], 
                              self.gen_conv_layers_strides[i],
                              dtype=dtype
                              ))
            # This self.gen_conv_crops logic is soooo annoying, but it's necessary in order to
            # match tensorflow's padding="SAME"
            smaller_crop = math.floor((pytorch_traj_len - desired_traj_len)/2)
            larger_crop = pytorch_traj_len - desired_traj_len - smaller_crop
            if smaller_crop + larger_crop > 0:
                self.gen_conv_crops.append(
                      torch.nn.ConstantPad1d((-smaller_crop, -larger_crop),0))
            else:
                self.gen_conv_crops.append(None)
            traj_len = desired_traj_len
            prev_channels = layer_channels

        self.gen_fcs = nn.ModuleList(self.gen_fcs)
        self.gen_convs = nn.ModuleList(self.gen_convs)

        self.nonlinearity = torch.nn.Softplus() if dec_use_softplus else (torch.nn.ELU() if dec_conv_use_elu else torch.nn.ReLU())

    def get_mean_square_layer_weights(self):
        sum_weights = 0
        count_weights = 0
        for layer in self.gen_convs:
            sum_weights += torch.sum(torch.square(layer.weight))
            count_weights += np.product(np.array(layer.weight.shape)).item()
        for layer in self.gen_fcs:
            sum_weights += torch.sum(torch.square(layer.weight))
            count_weights += np.product(np.array(layer.weight.shape)).item()
        return sum_weights/count_weights


    def decode(self, zs, ts):
      trajectory, _ = self.decode_and_return_noisy_embedding_node(zs,ts)
      return trajectory

    # given a batch of zs
    # you can completely ignore the ts
    def decode_and_return_noisy_embedding_node(self, zs, ts):
      layer = zs
      num_fcs = len(self.gen_fcs)
      num_convs = len(self.gen_convs)
      for i, fc in enumerate(self.gen_fcs):
          layer = fc(layer)
          # special logic to not do ReLu on last FC layer if we only have Fcs
          if not (num_convs == 0 and i+1 == num_fcs):
            layer = self.nonlinearity(layer)
      layer = layer.view(-1,
              self.gen_first_conv_channels,
              self.gen_first_traj_len)
      for i, conv in enumerate(self.gen_convs):
          layer = conv(layer)
          # special logic to not do ReLu on last conv layer 
          if not (i+1 == num_convs):
            layer = self.nonlinearity(layer)
          # This self.gen_conv_crops logic is soooo annoying, but it's necessary in order to
          # match tensorflow's padding="SAME"
          if self.gen_conv_crops[i] is not None:
              layer = self.gen_conv_crops[i](layer)

      # again, remember that CONV expects (batchsize, traj_channels, traj_len)
      # but all the rest of our code wants those last two switched.
      # so, switch them! 
      return(layer.transpose(1,2)), zs

# This is an (even more) standard 1D convolutional decoder
# used in 1D-CONVOLUTIONAL AUTOENCODER BASED HYPERSPECTRAL DATA COMPRESSION
# it uses upsampling instead of upconvolution to increase granularity
class OneDConvDecoderUpsampling(nn.Module):
    def __init__(self,
            latent_dim, 
            dec_gen_fc_layers_num_features = [16*32],
            dec_gen_first_traj_len=16,
            dec_gen_conv_layers_channels = [32, 32, 7],
            dec_gen_upsampling_factors = [2,2,1],
            dec_gen_conv_layers_kernel_sizes = [5,5,5],
            dec_use_softplus = False,
            dec_use_elu = False,
            dec_conv_use_elu = False,
            dec_use_tanh = False,
            dtype=torch.float,
            dec_initial_log_noise_estimate = None,
            **kwargs):
        super(OneDConvDecoderUpsampling, self).__init__()
        if dec_initial_log_noise_estimate is not None:
          dec_initial_log_noise_estimate = np.array(dec_initial_log_noise_estimate).item() # round-trip idempotently to just a number
          self.log_decoder_variance = nn.parameter.Parameter(data=torch.FloatTensor((dec_initial_log_noise_estimate,)), requires_grad=True)
        else:
          self.log_decoder_variance = torch.FloatTensor((1.,))
        self.latent_dim = latent_dim
        self.gen_fc_layers_num_features = dec_gen_fc_layers_num_features
        self.gen_first_traj_len= dec_gen_first_traj_len
        self.gen_conv_layers_channels = dec_gen_conv_layers_channels
        self.gen_upsampling_factors = dec_gen_upsampling_factors
        self.gen_conv_layers_kernel_sizes = dec_gen_conv_layers_kernel_sizes 
        # construct the parameters for all the generative fully-connected layers
        self.gen_fcs = []
        prev_features = self.latent_dim
        layer_features = prev_features
        for layer_features in self.gen_fc_layers_num_features:
            self.gen_fcs.append(nn.Linear(prev_features, layer_features,dtype=dtype))
            prev_features = layer_features
        # construct the parameters for all the generative convolutions
        traj_len = self.gen_first_traj_len
        self.gen_first_conv_channels = int(layer_features/traj_len )
        prev_channels = self.gen_first_conv_channels
        self.gen_convs = []
        for i, layer_channels in enumerate(self.gen_conv_layers_channels):
            self.gen_convs.append(
                    nn.Conv1d(prev_channels,
                              layer_channels, 
                              self.gen_conv_layers_kernel_sizes[i], 
                              stride = 1,
                              padding = "same",
                              dtype = dtype
                              ))
            prev_channels = layer_channels

        self.gen_fcs = nn.ModuleList(self.gen_fcs)
        self.gen_convs = nn.ModuleList(self.gen_convs)

        # A backward incompatible change, followed by training with use_tanh simultaneous with use_elu
        # led to the following unfortunate way to parse models in a backward-compatible fashion
        self.nonlinearity = torch.nn.Softplus() if dec_use_softplus else (
               torch.nn.Tanh() if (dec_use_tanh and not dec_use_elu) else (
               torch.nn.ELU() if ((dec_use_tanh and dec_use_elu) or dec_conv_use_elu) else 
               torch.nn.ReLU()))
        self.latent_dim = latent_dim

    def get_mean_square_layer_weights(self):
        sum_weights = 0
        count_weights = 0
        for layer in self.gen_convs:
            sum_weights += torch.sum(torch.square(layer.weight))
            count_weights += np.product(np.array(layer.weight.shape)).item()
        for layer in self.gen_fcs:
            sum_weights += torch.sum(torch.square(layer.weight))
            count_weights += np.product(np.array(layer.weight.shape)).item()
        return sum_weights/count_weights


    def decode(self, zs, ts):
      trajectory, _ = self.decode_and_return_noisy_embedding_node(zs,ts)
      return trajectory

    # given a batch of zs
    # you can completely ignore the ts
    def decode_and_return_noisy_embedding_node(self, zs, ts):
      # here, we add logic which is trivial/do nothing if
      # we're taking in a latent dim of the expected size.
      # However, if we're taking in a rate_invariant latent dim,
      # then we just want to take the last latent_dim values to decode
      # (this is only used if we're training a conv decoder on a rate_invariant
      # model). Not sure why we would ever do that.... but shh.
      layer = zs[:,-self.latent_dim:]
      num_fcs = len(self.gen_fcs)
      num_convs = len(self.gen_convs)
      for i, fc in enumerate(self.gen_fcs):
          layer = fc(layer)
          # special logic to not do ReLu on last FC layer if we only have Fcs
          if not (num_convs == 0 and i+1 == num_fcs):
            layer = self.nonlinearity(layer)
      layer = layer.view(-1,
              self.gen_first_conv_channels,
              self.gen_first_traj_len)
      for i, conv in enumerate(self.gen_convs):
          # CONV works on shapes (batchsize, traj_channels, traj_len)
          layer = torch.repeat_interleave(layer, self.gen_upsampling_factors[i], axis=2)
          layer = conv(layer)
          # special logic to not do ReLu on last conv layer 
          if not (i+1 == num_convs):
            layer = self.nonlinearity(layer)

      # again, remember that CONV expects (batchsize, traj_channels, traj_len)
      # but all the rest of our code wants those last two switched.
      # so, switch them! 
      return(layer.transpose(1,2)), zs

# for this API, for simplicity, latent_dim needs to be 
class RateInvariantDecoder(OneDConvDecoderUpsampling):
  def __init__(self, ria_T, **kwargs):
    super(RateInvariantDecoder, self).__init__(**kwargs)
    # the first T latent dimensions are actually the timewarping dims
    self.T = ria_T

  #overwrite this method to differently use the first self.T and last latent_dim variables
  def decode_and_return_noisy_embedding_node(self, mu, ts):
      # latents are last indices
      z = mu[:, self.T:]

      # mu is of shape (batch_size, T+d), so we break it apart into two separate tensors
      # the first, v is the timing parameters
      v = mu[:, :self.T]
      # the timing parameters are converted to gamma parameters by squaring and dividing by the sum of the squares
      gamma = v**2 / torch.sum(v**2, dim=1, keepdim=True)
      x, _ = super().decode_and_return_noisy_embedding_node(z, None)

      recon = pvtw.warp(x, gamma)
      return recon, z

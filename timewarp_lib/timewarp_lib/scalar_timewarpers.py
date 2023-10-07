import numpy as np
import torch
import torch.nn as nn

# currently we need embed_trajectory to learn the parameter encoder on the "Timewarper" object
import timewarp_lib.encoders as et

import timewarp_lib.utils.parametric_monotonic_function as pmf


# Do no scalar timewarping
class IdentityScalarTimewarper(nn.Module):
    def __init__(self, **kwargs):
        super(IdentityScalarTimewarper, self).__init__()

    def timewarp(self, xs, ts):
        return ts
    
    def get_parameters_from_poses(self, xs):
      raise Exception("not implemented")


# Do scalar timewarping based on parameters MODELED off of the trajectory (using OneDConvEncoder)
class ModeledParameterScalarTimewarper(nn.Module):
    def __init__(self, 
            traj_len,
            traj_channels,
            scaltw_granularity,
            scaltw_emb_conv_layers_channels = [],
            scaltw_emb_conv_layers_strides = [],
            scaltw_emb_conv_layers_kernel_sizes = [],
            scaltw_emb_fc_layers_num_features = [],
            scaltw_min_canonical_time = 0.,
            scaltw_max_canonical_time = 1.,
            dtype=torch.float,
            **kwargs):
        super(ModeledParameterScalarTimewarper, self).__init__()

        # I know, this is horrifying...but at the same time...brilliant
        # we just ignore the part of this architecture that outputs fclogvar
        # and instead just use the fcmu component as our scalar timewarping parameters
        self.timewarp_parameter_encoder = et.OneDConvEncoder(latent_dim=scaltw_granularity,
          traj_len = traj_len,
          traj_channels = traj_channels,
          emb_conv_layers_channels = scaltw_emb_conv_layers_channels,
          emb_conv_layers_strides = scaltw_emb_conv_layers_strides,
          emb_conv_layers_kernel_sizes = scaltw_emb_conv_layers_kernel_sizes,
          emb_fc_layers_num_features = scaltw_emb_fc_layers_num_features)

        # The easiest way to make the timewarping function be a bijection on [minT,maxT]
        # is to use LogSoftmax
        self.granularity = scaltw_granularity
        # when loading from a saved file, scaltw_min and max params will be numpy
        # the following is an idempotent way to convert to non-numpy array numbers
        self.scaltw_min_canonical_time = np.array(scaltw_min_canonical_time).item()
        self.scaltw_max_canonical_time = np.array(scaltw_max_canonical_time).item()
        self.LogSoftmaxLayer = torch.nn.LogSoftmax(dim=1)

        # This is not learnable, but is instead just a cached function that is the correct
        # size to apply our self.time_transform_coeffs
        self.monotonic_applier = pmf.ParameterizedMonotonicApplier(granularity = scaltw_granularity,
                                                                   dtype=dtype)

        # The initialization here should set everything to zero to start
        # This model class matches Encoders.OneDConvEncoder so it has
        # an fcmu Linear Layer we can set to zero
        with torch.no_grad():
              self.timewarp_parameter_encoder.fcmu.bias.zero_()
              self.timewarp_parameter_encoder.fcmu.weight.zero_()

    # commented out, since I think I fixed this by making monotonic_applier a nn.Module 
    #def to(self, device):
    #    print(f"Sending Timewarper to {device}")
    #    self = super(Timewarper, self).to(device)
        # monotonic_applier isn't an nn.Module,
        # so we have to explicitly move it to GPU (TODO why not just make it an nn.Module?)
    #    self.monotonic_applier = self.monotonic_applier.to(device)
    #    return self

    def get_parameters_from_poses(self, xs):
        # ignore the "logvar" part
        time_transform_coeffs, _ = self.timewarp_parameter_encoder.encode(xs)
        # new: take the LogSoftmax so that Exp of the time_transform_coeffs sums to exactly self.granularity
        # (ie: so that the slope has mean 1)
        time_transform_coeffs = self.LogSoftmaxLayer(time_transform_coeffs) + np.log(self.granularity) 
        return time_transform_coeffs

    def timewarp(self, xs, ts):
        if not len(ts.shape)==3 or not ts.shape[2] == 1:
          raise Exception("ts should be a (batchsize,timesteps,channel=1) matrix")
        time_transform_coeffs = self.get_parameters_from_poses(xs)
        # super easy way to take reasonable coefficients but then scale nicely. We like this
        # because the scaling penalty (based on time_transform_coeffs) doesn't need to change. yay.
        scaled_ts = (self.monotonic_applier.batch_apply_monotonic_transformation(time_transform_coeffs,ts) * (self.scaltw_max_canonical_time-self.scaltw_min_canonical_time) 
                     + self.scaltw_min_canonical_time)
        return scaled_ts

###
### An interesting idea, but currenlty not implemented/used (not curently passing around training_id)
###
## Do scalar timewarping based on parameters LOOKED UP from a table (based on training trajectory_id)
#class TabularScalarTimewarper(nn.Module):
#    def __init__(self, training_count, granularity,
#                 dtype=torch.float):
#        super(TabularScalarTimewarper, self).__init__()
#
#        # table to look up coefficients
#        self.time_transform_coeffs = torch.nn.ParameterList(
#            [nn.parameter.Parameter(torch.tensor(np.zeros(shape=(1,granularity)),dtype=dtype))
#             for _ in range(training_count)])
#
#        # This is not learnable, but is instead just a cached function that is the correct
#        # size to apply our self.time_transform_coeffs
#        self.monotonic_applier = pmf.ParameterizedMonotonicApplier(granularity = granularity,
#                                                                   dtype=dtype)
#
#    # commented out, since I think I fixed this by making monotonic_applier a nn.Module 
#    #def to(self, device):
#    #    self = super(TabularScalarTimewarper, self).to(device)
#    #    # monotonic_applier isn't an nn.Module,
#    #    # so we have to explicitly move it to GPU
#    #    self.monotonic_applier = self.monotonic_applier.to(device)
#    #    return self
#
#    # look up the transform coeffs from the associated table
#    def get_parameters_from_training_index(self, training_index):
#      if not np.isscalar(training_index) or int(training_index) - training_index != 0:
#        raise Exception("training_index must be an integer")
#      time_transform_coeffs = self.time_transform_coeffs[training_index]
#      return time_transform_coeffs
#
#    # training timewarp can only be applied to the set of training timestamps
#    # since it requires looking up trajectory parameters
#    def timewarp(self, xs, training_index):
#      if not len(ts.shape)==3 or not ts.shape[0] == 1 or not ts.shape[2] == 1:
#        raise Exception("ts should be a (batchsize=1,timesteps,channel=1) matrix")
#      time_transform_coeffs = self.get_parameters_from_training_index(training_index)
#      scaled_ts = self.monotonic_applier.batch_apply_monotonic_transformation(time_transform_coeffs,ts)
#      # add back in that first dimension we removed above
#      return scaled_ts

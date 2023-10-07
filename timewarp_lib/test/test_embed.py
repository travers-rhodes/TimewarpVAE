import unittest
import torch
import numpy as np
import timewarp_lib.encoders as et

def get_xs_ts(batch_size, traj_len, traj_channels):
  xs = torch.rand((batch_size, traj_len, traj_channels))
  scaled_ts = torch.rand((batch_size, traj_len, 1))
  return xs,scaled_ts

class TestEncode(unittest.TestCase):
  def test_transform_encode(self):
    latent_dim=3
    traj_len=13
    traj_channels=7
    attention_dims_per_head=5
    embedder = et.TransformerEncoder(latent_dim, traj_len, traj_channels, attention_dims_per_head)

    
    for batch_size in [49, 101]:
      xs, scaled_ts = get_xs_ts(batch_size, traj_len, traj_channels)
      mus, logvars = embedder.encode(xs, scaled_ts)
      np.testing.assert_equal(mus.shape, (batch_size, latent_dim))
      np.testing.assert_equal(logvars.shape, (batch_size, latent_dim))
  
  def test_conv_encode(self):
    latent_dim=3
    traj_len=13
    traj_channels=7
    embedder = et.OneDConvEncoder(latent_dim, traj_len, traj_channels)

    
    for batch_size in [49, 101]:
      xs, scaled_ts = get_xs_ts(batch_size, traj_len, traj_channels)
      mus, logvars = embedder.encode(xs, scaled_ts)
      np.testing.assert_equal(mus.shape, (batch_size, latent_dim))
      np.testing.assert_equal(logvars.shape, (batch_size, latent_dim))



if __name__ == '__main__':
  unittest.main()


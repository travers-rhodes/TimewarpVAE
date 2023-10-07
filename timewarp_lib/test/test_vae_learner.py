import unittest
import torch
import numpy as np
import timewarp_lib.encoders as et
import timewarp_lib.decoders as dt
import timewarp_lib.scalar_timewarpers as tim
import timewarp_lib.vae_template as vl

def get_xs_ts(batch_size, traj_len, traj_channels, dtype=torch.float):
  xs = torch.rand((batch_size, traj_len, traj_channels),dtype=dtype)
  scaled_ts = torch.rand((batch_size, traj_len, 1),dtype=dtype)
  return xs,scaled_ts

class TestVAELearner(unittest.TestCase):
  def test_transformer_vae_learner(self):
    for dtype in [torch.float, torch.double]:
      latent_dim=3
      traj_len=13
      traj_channels=7
      attention_dims_per_head=5
      embedder = et.TransformerEncoder(latent_dim, traj_len, traj_channels, attention_dims_per_head,dtype=dtype)
      decoder = dt.FunctionStyleDecoder(latent_dim, traj_channels,dtype=dtype)

      granularity = 53
      for batch_size in [49, 101]:
        training_count = batch_size
        timewarper = tim.IdentityScalarTimewarper() 
        hi = vl.VAE(embedder, decoder, timewarper)
        xs, ts = get_xs_ts(batch_size, traj_len, traj_channels,dtype=dtype)
        for tid in range(batch_size):
          x = xs[tid:tid+1]
          t = ts[tid:tid+1]
          xout, mu, logvar, z, scaled_ts = hi.forward(x,t)
          np.testing.assert_equal(xout.shape, (1,traj_len, traj_channels))
          np.testing.assert_equal(scaled_ts.shape, (1,traj_len, 1))
          xout, _, _, scaled_ts = hi.noiseless_forward(x,t)
          np.testing.assert_equal(xout.shape, (1,traj_len, traj_channels))
          np.testing.assert_equal(scaled_ts.shape, (1,traj_len, 1))
  
  def test_conv_embed(self):
    for dtype in [torch.float, torch.double]:
      latent_dim=3
      traj_len=13
      traj_channels=7
      embedder = et.OneDConvEncoder(latent_dim, traj_len, traj_channels,dtype=dtype)
      decoder = dt.FunctionStyleDecoder(latent_dim, traj_channels, dtype=dtype)

      granularity = 53
      for batch_size in [49, 101]:
        training_count = batch_size
        timewarper = tim.IdentityScalarTimewarper() 
        hi = vl.VAE(embedder, decoder, timewarper)
        xs, ts = get_xs_ts(batch_size, traj_len, traj_channels, dtype=dtype)
        for tid in range(batch_size):
          x = xs[tid:tid+1]
          t = ts[tid:tid+1]
          xout, mu, logvar, z, scaled_ts = hi.forward(x,t)
          np.testing.assert_equal(xout.shape, (1,traj_len, traj_channels))
          np.testing.assert_equal(scaled_ts.shape, (1,traj_len, 1))
          xout, _, _, scaled_ts = hi.noiseless_forward(x,t)
          np.testing.assert_equal(xout.shape, (1,traj_len, traj_channels))
          np.testing.assert_equal(scaled_ts.shape, (1,traj_len, 1))
  
  def test_conv_encode_conv_decode_batch(self):
    for dtype in [torch.float,torch.double]:
      latent_dim=3
      traj_len=45
      traj_channels=7
      embedder = et.OneDConvEncoder(latent_dim, traj_len, traj_channels,dtype=dtype)
      decoder = dt.OneDConvDecoder(
          latent_dim,
          dtype=dtype,
     dec_gen_fc_layers_num_features = [15],
     dec_gen_first_traj_len=15,
     dec_gen_conv_layers_channels = [7],
     # first_traj_len times all strides should give final size
     dec_gen_conv_layers_strides = [3],
     dec_gen_conv_layers_kernel_sizes = [5],
     dec_use_softplus=True,
          )

      granularity = 53
      for batch_size in [49, 101]:
        training_count = batch_size
        timewarper = tim.IdentityScalarTimewarper()
        hi = vl.VAE(embedder, decoder, timewarper)
        xs, ts = get_xs_ts(batch_size, traj_len, traj_channels, dtype=dtype)
        # run it all as a single batch
        tid = None
        x = xs
        t = ts
        xout, mu, logvar, z, scaled_ts = hi.forward(x,t)
        #expected_traj_len = (traj_len + 1) * self.gen_conv_layers_strides[i] - self.gen_conv_layers_kernel_sizes[i]
        expected_traj_len = (15+ 1) * 3 - 5
        np.testing.assert_equal(xout.shape, (batch_size,expected_traj_len, traj_channels))
        np.testing.assert_equal(scaled_ts.detach().cpu().numpy(), t.detach().cpu().numpy())
        xout, _, _, scaled_ts = hi.noiseless_forward(x,t)
        np.testing.assert_equal(xout.shape, (batch_size,expected_traj_len, traj_channels))
        np.testing.assert_equal(scaled_ts.detach().cpu().numpy(), t.detach().cpu().numpy())
  
  def test_conv_decode(self):
    for dtype in [torch.float]:
      latent_dim=3
      traj_len=45
      traj_channels=7
      embedder = et.OneDConvEncoder(latent_dim, traj_len, traj_channels,dtype=dtype)
      decoder = dt.OneDConvDecoder(latent_dim,
     dec_gen_fc_layers_num_features = [15],
     dec_gen_first_traj_len=15,
     dec_gen_conv_layers_channels = [7],
     # first_traj_len times all strides should give final size
     dec_gen_conv_layers_strides = [3],
     dec_gen_conv_layers_kernel_sizes = [5],
     dec_use_softplus=True,
     )

      granularity = 53
      for batch_size in [49, 101]:
        training_count = batch_size
        timewarper = tim.IdentityScalarTimewarper() 
        hi = vl.VAE(embedder, decoder, timewarper)
        xs, ts = get_xs_ts(batch_size, traj_len, traj_channels, dtype=dtype)
        for tid in range(batch_size):
          x = xs[tid:tid+1]
          t = ts[tid:tid+1]
          xout, mu, logvar, z, scaled_ts = hi.forward(x,t)
          #expected_traj_len = (traj_len + 1) * self.gen_conv_layers_strides[i] - self.gen_conv_layers_kernel_sizes[i]
          expected_traj_len = (15+ 1) * 3 - 5
          np.testing.assert_equal(xout.shape, (1,expected_traj_len, traj_channels))
          np.testing.assert_equal(scaled_ts.shape, (1,traj_len, 1))
          xout, _, _, scaled_ts = hi.noiseless_forward(x,t)
          np.testing.assert_equal(xout.shape, (1,expected_traj_len, traj_channels))
          np.testing.assert_equal(scaled_ts.shape, (1,traj_len, 1))
    

if __name__ == '__main__':
  unittest.main()


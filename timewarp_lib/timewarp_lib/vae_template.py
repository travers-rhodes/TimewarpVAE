import torch
import torch.nn as nn
import numpy as np

def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn((mu.shape[0], mu.shape[1]), device=mu.device)
        return mu + eps*std

class VAE(nn.Module):
    def __init__(self, encoder, decoder, scalar_timewarper, spherical_noise_instead_of_diagonal = False):
        super(VAE, self).__init__()

        # The following parameters are all learnable modules
        self.encoder = encoder
        self.decoder = decoder
        self.scalar_timewarper = scalar_timewarper

        self.spherical_noise_instead_of_diagonal = spherical_noise_instead_of_diagonal

    # forward but don't add any of the encoder noise
    def noiseless_forward(self, xs, ts):
      assert len(ts.shape) == 3, "ts of shape (batch_size, traj_len, 1)"
      t_bsize, t_tlen, t_tdim = ts.shape
      assert t_tdim == 1, "third dimension of ts should have size 1"
      assert len(xs.shape) == 3, "for consistency with other apis, xs should include batch_size and be shape (batch_size, traj_len, traj_dim)"
      bsize, tlen, tdim = xs.shape

      assert t_bsize == bsize and t_tlen == tlen, "xs dims should match ts dims"

      scaled_ts = self.scalar_timewarper.timewarp(xs, ts)
      # the encode function assumes xs and ts both have shape (batchsize, traj_len, X), so we unsqueeze ts
      mu, logvar = self.encoder.encode(xs,scaled_ts)
      x = self.decoder.decode(mu, scaled_ts)
      return x, mu, logvar, scaled_ts

    # standard VAE forward with encoder noise
    def forward_with_noisy_embedding_node(self, xs, ts):
      assert len(ts.shape) == 3, "ts of shape (batch_size, traj_len, 1)"
      t_bsize, t_tlen, t_tdim = ts.shape
      assert t_tdim == 1, "third dimension of ts should have size 1"
      assert len(xs.shape) == 3, "for consistency with other apis, xs should include batch_size and be shape (batch_size, traj_len, traj_dim)"
      bsize, tlen, tdim = xs.shape

      assert t_bsize == bsize and t_tlen == tlen, "xs dims should match ts dims"

      scaled_ts = self.scalar_timewarper.timewarp(xs, ts)
      mu, logvar = self.encoder.encode(xs,scaled_ts)
      if self.spherical_noise_instead_of_diagonal:
        # only use the first logvar---ignore the rest (they'll be broadcast copied from the 1st)
        logvar = logvar[:,:1]
      z = reparameterize(mu,logvar)
      x, broadcast_zs = self.decoder.decode_and_return_noisy_embedding_node(z, scaled_ts)
      return x, mu, logvar, z, scaled_ts, broadcast_zs
    
    def forward(self, xs, ts):
      x, mu, logvar, z, scaled_ts, _ = self.forward_with_noisy_embedding_node(xs, ts)
      return x, mu, logvar, z, scaled_ts

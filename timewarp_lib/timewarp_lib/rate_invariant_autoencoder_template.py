import torch
import torch.nn as nn
import timewarp_lib.parameterized_vector_time_warper as pvtw

# Reimplementation of the RateInvariantAutoencoder from
# K. Koneripalli, S. Lohit, R. Anirudh and P. Turaga, "Rate-Invariant Autoencoding of Time-Series," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 3732-3736, doi: 10.1109/ICASSP40776.2020.9053983.
class RateInvariantAutoencoder(nn.Module):
  # Slight redefinition here compared to the original paper:
  # T is the number of time steps in the output MINUS ONE.
  # That's because T should be the number of intervals and there is
  # one fewer interval than there are data points.
  # d is the spatial latent encoding dimension
  def __init__(self, encoder, decoder, ria_T, latent_dim, **kwargs):
        super(RateInvariantAutoencoder, self).__init__()

        # The following parameters are all learnable modules
        self.encoder = encoder
        self.decoder = decoder
        self.T = ria_T
        print("This is a tuple?",self.T)
        self.d = latent_dim

  def noiseless_forward(self, xs):
      assert len(xs.shape) == 3, "for consistency with other apis, xs should include batch_size and be shape (batch_size, traj_len, traj_dim)"

      # for consistency with other api's we pass timestamps of None (unused)
      # ignore the "logvar" output of the encoder (rather than refactoring all our encoders to not return a logvar, just ignore it)
      mu, _ = self.encoder.encode(xs,None)
      print("mu.shape is ", mu.shape)
      # mu is of shape (batch_size, T+d), so we break it apart into two separate tensors
      # the first, v is the timing parameters
      v = mu[:, :self.T]
      # the timing parameters are converted to gamma parameters by squaring and dividing by the sum of the squares
      gamma = v**2 / torch.sum(v**2, dim=1, keepdim=True)
      # the second, z is the spatial latent embedding that we actually want to decode
      z = mu[:, self.T:]
      x = self.decoder.decode(z, None)

      recon = pvtw.warp(x, gamma)
      return recon, mu, z, gamma, x

  def forward(self, xs):
      recon, mu, z, gamma, x = self.noiseless_forward(xs)
      return recon, mu, z, gamma, x

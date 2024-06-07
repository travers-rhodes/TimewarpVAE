import torch

# the warp function takes in
# x: the input trajectory of shape (batch_size, traj_len, traj_dim)
# gammadot: the warping parameters of shape (batch_size, T)
#   gammadot should already be normalized to sum to 1.
#   We'll multiply it by T in this function so that it sums to T
#   and so that its values can be used to index into a vector of length T+1
#
# In the original TTN (https://arxiv.org/pdf/1906.05947.pdf) work they had
# "The network outputs a vector of length T, such that the first element is set to be zero"
# Ok...but instead we're going to ignore that zero and just say that our T is one fewer
# than the original paper's T.
# 
# maps them through to output timestamps according to the warping function
# defined by having slope gamma at each interval [0,1/T], [1/T, 2/T], ..., [(T-1)/T, 1]
# be equal to the input values gammadot
#
# This implementation is informed by https://github.com/suhaslohit/TTN
def warp(x,gammadot):
  assert len(x.shape) == 3, "x of shape (batch_size, traj_len, traj_dim)"
  assert len(gammadot.shape) == 2, "gammadot of shape (batch_size, T)"
  bsize, tlen, nchannels = x.shape
  t_bsize, T = gammadot.shape
  assert bsize == t_bsize, "batch sizes should match"
  assert tlen == T + 1, f"traj_len should be one more than T, but the sizes were tlen:{tlen} and T:{T}"

  # gamma are the desired time sampling for each of the T+1 datapoints
  gamma = torch.zeros((bsize, T+1), dtype=torch.float).to(device=x.device)
  gamma[:,1:] = torch.cumsum(gammadot, dim=1) * T
  # left samples are the samples at the left edge of each interval
  # they will be indexed by l_ind
  left_samples = torch.floor(gamma).long()
  left_samples = torch.clamp(left_samples, min=0, max=T-1)
  right_samples = left_samples + 1

  gamma_tiled = gamma.unsqueeze(2).expand(-1,-1,nchannels)
  left_samples_tiled = left_samples.unsqueeze(2).expand(-1,-1,nchannels)
  right_samples_tiled = right_samples.unsqueeze(2).expand(-1,-1,nchannels)

  # The relative weights for the left and right samples for linear interpolation
  w1 = gamma_tiled - left_samples_tiled
  w0 = 1 - w1

  left_samples_x = torch.gather(x, dim=1, index=left_samples_tiled)
  right_samples_x = torch.gather(x, dim=1, index=right_samples_tiled)

  return w0 * left_samples_x + w1 * right_samples_x







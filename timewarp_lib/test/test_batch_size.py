import torch
import torch.nn
import numpy as np
import pytest
import time


#import numpy_dtw as ndtw
import cpp_dtw as cdtw
#import cpp_dtw_cuda_kernel as cdtwcuda
#import cpp_dtw_cuda_kernel_split as cdtwcudasplit

from torch.profiler import profile, record_function, ProfilerActivity

### Discrete time warping (ie: many-to-many matches of closest)
###   no constraints
###   mean over batch 
###   integrated over dt_actual timesteps (not symmetric) (like a mean)
###   summed over num_channels
def tslearn_dtw_actual_weighting(recon, actual):
  import tslearn.metrics
  dtype = recon.dtype
  device = recon.device
  first_numpy = recon.detach().cpu().numpy()
  second_numpy = actual.detach().cpu().numpy()
  batch_size = first_numpy.shape[0]
  recon_time_steps = first_numpy.shape[1]
  actual_time_steps = second_numpy.shape[1]
  num_channels = first_numpy.shape[2]

  #####
  ##### Warping Matrix Definition
  #####
  ##### the warping matrix is of shape (batch_size, actual_time_steps, recon_time_steps)
  ##### the sum over recon_time_steps is 1 for each pair (batch_size, actual_time_steps)
  ##### and the value in each of the non-zero cells in that sum is equal. 
  ##### 
  warp_matrix = np.zeros((batch_size, actual_time_steps, recon_time_steps))

  # copied from naive implementation
  timewarped_first = np.zeros(second_numpy.shape)
  # For the first, simplest, trivialest test implementation, model each time series
  # separately 
  for i in range(batch_size):
    # for this time series, compute the path
    # that warps the two time series to be as similar as possible
    path, dist = tslearn.metrics.dtw_path(first_numpy[i], second_numpy[i])
    path = np.array(path)
    for actual_ind in range(len(second_numpy[i])):
      # get all the matches to that reconstruction index
      pair_inds = path[path[:,1]==actual_ind,:]
      weight = 1/pair_inds.shape[0]
      for pair_ind in pair_inds:
        warp_matrix[i,pair_ind[1],pair_ind[0]] = weight

  warp_matrix_tensor = torch.tensor(warp_matrix, dtype=dtype).to(device)

  warped_recon = torch.einsum("bar,brc->bac",warp_matrix_tensor,recon)
  tslearnloss = torch.nn.functional.mse_loss(warped_recon, actual, reduction='mean') * num_channels
  return tslearnloss

def make_recon_actual(device, length, batch_size):
  error = 0.1
  dtype = torch.float
  num_channels = 7
  np_actual = np.arange(length*batch_size*num_channels,dtype=float).reshape(batch_size,length,num_channels)
  np_recon = np_actual + error
  batch_size = np_recon.shape[0]
  num_timesteps = np_recon.shape[1]

  recon = torch.tensor(np_recon, dtype=dtype).to(device)
  # sending to(device) resets requires_grad
  recon.requires_grad_(True)
  actual = torch.tensor(np_actual, dtype=dtype).to(device)

  return (recon,actual)

def simple_apply_and_check(function_name, recon, actual, device="cpu"):
  #if recon.grad is not None:
  #  recon.grad.zero_();
  if device != "cpu":
    torch.cuda.synchronize();
  start = time.time()
  loss = function_name(recon, actual)
  if device != "cpu":
    torch.cuda.synchronize();
  end = time.time()
  return (end-start)

@pytest.mark.filterwarnings("ignore:")
def test_numpy_dtw_actual_weighting():
  for length in [1000]:
    for batch_size in [100]:
      recon, actual = make_recon_actual("cpu",length,batch_size)
      simple_apply_and_check(cdtw.dtw_loss, recon, actual)
      timing = simple_apply_and_check(cdtw.dtw_loss, recon, actual)
      print(f"{length} {batch_size} cdtw",timing)

      #recon, actual = make_recon_actual("cuda",length,batch_size)
      #simple_apply_and_check(cdtwcudasplit.dtw_loss, recon, actual)
      #timing = simple_apply_and_check(cdtwcudasplit.dtw_loss, recon, actual)
      #print(f"{length},{batch_size} cdtwcudasplit",timing)

      #simple_apply_and_check(cdtwcuda.dtw_loss,recon,actual)
      #timing = simple_apply_and_check(cdtwcuda.dtw_loss,recon,actual)
      #print(f"{length},{batch_size} cdtwcuda",timing)


if __name__=="__main__":
    test_numpy_dtw_actual_weighting()


import torch

import torch.nn
import numpy as np
import pytest
import time


#import numpy_dtw as ndtw
#import cpp_linear_dtw as cldtw
#import cpp_linear_dtw_cuda_kernel as cldtwcuda
#import cpp_approx_linear_dtw_cuda_kernel as caldtwcuda
#import numpy_linear_dtw as nldtw
#import numpy_linear_dtw_njit as nldtwnjit
import cpp_dtw as cdtw
#import cpp_dtw_cuda_kernel as cdtwcuda

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

def make_recon_actual(device, big_length, small_length, batch_size):
  error = 0.1
  dtype = torch.float
  num_channels = 7
  np_actual = np.random.normal(size=(big_length*batch_size*num_channels)).reshape(batch_size,big_length,num_channels)
  np_recon= np.random.normal(size=(small_length*batch_size*num_channels)).reshape(batch_size,small_length,num_channels)

  recon = torch.tensor(np_recon, dtype=dtype).to(device)
  # sending to(device) resets requires_grad
  recon.requires_grad_(True)
  actual = torch.tensor(np_actual, dtype=dtype).to(device)

  return (recon,actual)

def simple_apply_and_check(function_name, recon, actual, device, reg=False):
  #if recon.grad is not None:
  #  recon.grad.zero_();
  if device != "cpu":
    torch.cuda.synchronize();
  start = time.time()
  if reg:
    regularization_lambda = float(0.0)
    loss = function_name(recon, actual, regularization_lambda)
  else:
    loss = function_name(recon, actual)
  if device != "cpu":
    torch.cuda.synchronize();
  end = time.time()
  return (end-start)

@pytest.mark.filterwarnings("ignore:")
def test_all_timings():
  timing_results = []
  for big_size in [16,32]:#,64,125,250,500,1000]:
    for small_size in [16,32]:#,64,1000,125,250,500,100]:
      for batch_size in [16,32]:#,64,125,250,500,1000,2000]:
        if small_size > big_size:
          continue
        for device, func, name, reg in [
            #("cpu", tslearn_dtw_actual_weighting, 0, False),
            ("cpu", cdtw.dtw_loss, 1, False),
            #("cpu", cldtw.dtw_loss, 2, True),
            #("cuda", cldtwcuda.dtw_loss, 3, True),
            #("cuda", cdtwcuda.dtw_loss, 4, False),
            #("cuda", caldtwcuda.dtw_loss, 5, True)
            ]:
          recon, actual = make_recon_actual(device, big_size, small_size, batch_size)
          timing = simple_apply_and_check(func, recon, actual, device, reg=reg)
          timing_results.append((name, batch_size, big_size, small_size, timing))
          print(timing_results[-1])
      #np.save("timing_results.npy", np.array(timing_results))

if __name__=="__main__":
  test_all_timings()
  #test_numpy_linear_dtw()


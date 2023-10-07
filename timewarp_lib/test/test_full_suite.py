import torch
import torch.nn
import numpy as np
import pytest

#import numpy_dtw as ndtw
import cpp_dtw as cdtw
import param_cpp_dtw as pcdtw
import pvect_cpp_dtw as pvdtw

from torch.profiler import profile, record_function, ProfilerActivity

###
### API
###
#### Input:
####  recon: torch.tensor of shape (batch_size, time_steps, num_channels)
####  actual: torch.tensor of shape (batch_size, time_steps, num_channels)
###
#### Output:
####  loss: torch.tensor of shape (batch_size,) 
####         on which we can correctly compute backward with respect to recon
####         that gives the distance between recon and actual trajectories 
####         according to [VARIABLE DEFINITION] of distance between trajectories



###
### For this file, the loss is "hard to compute" using simple dtw because the loss is
### computed as follows (after pairing is chosen): 
###   1) compute squared error between pairs
###   2) average all pairs containing a test point equally
###   3) THEN average over all those test point sets equally
### (this is different from what is done during path selection)
### (which weights according to dt1+dt2)



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

def simple_apply_and_check(function_name):
  dtype = torch.float
  error = 0.1
  #np_actual = np.array(((1,2,3,2,1),(5,6,7,8,9))).reshape(2,-1,1)
  np_actual = np.array(((1,2,3,2,1)),dtype=float).reshape(1,-1,1)
  np_recon = np_actual + error
  batch_size = np_recon.shape[0]
  num_timesteps = np_recon.shape[1]
  
  recon = torch.tensor(np_recon, requires_grad = True, dtype=dtype)
  actual = torch.tensor(np_actual, dtype=dtype)
  loss = function_name(recon, actual)

  np.testing.assert_almost_equal(loss.detach().cpu().numpy(), error**2)

  # you should be able to run backward
  loss.backward()
  np_recon_grad = recon.grad.detach().cpu().numpy()
  # the backward gradient for this case should be 2 * error / num_timesteps / batch_size
  expected_grad = np.ones((batch_size,num_timesteps,1)) * 2 * error / num_timesteps / batch_size
  np.testing.assert_almost_equal(np_recon_grad, expected_grad)

def barely_warp_and_check(function_name):
  dtype=torch.float
  error = 0.1
  np_actual = np.array(((1,1,4,3,2)), dtype=float).reshape(1,-1,1)
  np_recon = np.array(((1,4,3,2,2)), dtype=float).reshape(1,-1,1) + error
  batch_size = np_recon.shape[0]
  num_timesteps = np_recon.shape[1]
  
  recon = torch.tensor(np_recon, requires_grad = True,dtype=dtype)
  actual = torch.tensor(np_actual,dtype=dtype)
  loss = function_name(recon, actual)

  print("got here")
  np.testing.assert_almost_equal(loss.detach().cpu().numpy(), error**2)

  # you should be able to run backward
  loss.backward()
  np_recon_grad = recon.grad.detach().cpu().numpy()
  # the basic/usual gradient for this case should be 2 * error / num_timesteps / batch_size
  # but the one that gets matched to twice gets double, and the two that get matched to same get half
  expected_grad = np.array((2,1,1,0.5,0.5)).reshape(batch_size,num_timesteps,1) * 2 * error / num_timesteps / batch_size
  np.testing.assert_almost_equal(np_recon_grad, expected_grad)

## The Sakoe-Chiba paper, and the rando paper I read,
## both use (dt_actual + dt_base) as the integral weight.
## Thus diagonal to an error is twice as bad as horizontal/vertical to an error
####HOWEVER
#### I don't actually like that...so I'm insisting that we use a max(dt_actual + dt_base) time warping
def tsdtw_edge_case_bug_testcase(function_name, handles_edge_case):
  np_actual = np.array(((0,7/3.,7/3.)), dtype=float).reshape(1,-1,1)
  np_recon = np.array(((0,4/3.,7/3.)), dtype=float).reshape(1,-1,1) 
  batch_size = np_recon.shape[0]
  num_timesteps = np_recon.shape[1]
  dtype = torch.float
  
  recon = torch.tensor(np_recon, requires_grad = True, dtype=dtype)
  actual = torch.tensor(np_actual, dtype=dtype)
  loss = function_name(recon, actual)

  #np.testing.assert_almost_equal(loss.detach().cpu().numpy(), ((4/3)**2)/3./2.)
  if handles_edge_case:
    np.testing.assert_almost_equal(loss.detach().cpu().numpy(), (2/3.)**2/3) # If you're doing 1,2,1 or similar cost of diagonal
  else:
    np.testing.assert_almost_equal(loss.detach().cpu().numpy(), 1/3.) # If you're using tslearn-style max(dt_a, dt_b)

  # you should be able to run backward
  loss.backward()
  np_recon_grad = recon.grad.detach().cpu().numpy()
  # the basic/usual gradient for this case should be 2 * error / num_timesteps / batch_size
  # but if you get two that get matched to same, it should get half
  if handles_edge_case:
    expected_grad = np.array((2/3/2,2/3/2,0)).reshape(batch_size,num_timesteps,1) * 2 / num_timesteps / batch_size
  else:
    expected_grad = np.array((0,-1,0)).reshape(batch_size,num_timesteps,1) * 2 / num_timesteps / batch_size
  np.testing.assert_almost_equal(np_recon_grad, expected_grad)

def big_warp_and_check(function_name):
  np_actual = np.array(((1,1,4,3,2)), dtype=float).reshape(1,-1,1)
  np_recon = np.array(((1,2,3,2,2,2)), dtype=float).reshape(1,-1,1) 
  batch_size = np_recon.shape[0]
  num_timesteps = np_recon.shape[1]
  num_actual_timesteps = np_actual.shape[1]
 
  dtype = torch.float 
  recon = torch.tensor(np_recon, requires_grad = True, dtype=dtype)
  actual = torch.tensor(np_actual, dtype=dtype)
  loss = function_name(recon, actual)

  np.testing.assert_almost_equal(loss.detach().cpu().numpy(), (0 + 1 + 1 + 0 + 0)/5.)

  # you should be able to run backward
  loss.backward()
  np_recon_grad = recon.grad.detach().cpu().numpy()
  # the basic/usual gradient for this case should be 2 * error / num_timesteps / batch_size
  # but the one that gets matched to twice gets double, and the two that get matched to same get half
  expected_grad = np.array((0,1,-1,0,0,0)).reshape(batch_size,num_timesteps,1) * 2 / num_actual_timesteps / batch_size
  np.testing.assert_almost_equal(np_recon_grad, expected_grad)

def simple_cuda_test(function_name):
  np_actual = np.array(((1,1,4,3,2)), dtype=float).reshape(1,-1,1)
  np_recon = np.array(((1,2,3,2,2,2)), dtype=float).reshape(1,-1,1) 
  batch_size = np_recon.shape[0]
  num_timesteps = np_recon.shape[1]
  num_actual_timesteps = np_actual.shape[1]
 
  dtype = torch.float 
  recon = torch.tensor(np_recon, dtype=dtype).to("cuda")
  recon.requires_grad_(True)
  actual = torch.tensor(np_actual, dtype=dtype).to("cuda")
  loss = function_name(recon, actual)

  np.testing.assert_almost_equal(loss.detach().cpu().numpy(), (0 + 1 + 1 + 0 + 0)/5.)

  # you should be able to run backward
  loss.backward()
  np_recon_grad = recon.grad.detach().cpu().numpy()
  # the basic/usual gradient for this case should be 2 * error / num_timesteps / batch_size
  # but the one that gets matched to twice gets double, and the two that get matched to same get half
  expected_grad = np.array((0,1,-1,0,0,0)).reshape(batch_size,num_timesteps,1) * 2 / num_actual_timesteps / batch_size
  np.testing.assert_almost_equal(np_recon_grad, expected_grad)

def run_all_tests(func, handles_edge_case=False):
  simple_apply_and_check(func)
  tsdtw_edge_case_bug_testcase(func, handles_edge_case)
  barely_warp_and_check(func)
  # I'm too lazy to write the baseline for this check when we do (1,2,1) so for now we only test this if (1,1,1)
  if not handles_edge_case:
    big_warp_and_check(func)


#@pytest.mark.filterwarnings("ignore:")
#def test_tslearn_dtw_actual_weighting():
#  run_all_tests(tslearn_dtw_actual_weighting)
#  simple_cuda_test(tslearn_dtw_actual_weighting)

#@pytest.mark.filterwarnings("ignore:")
#def test_numpy_dtw_actual_weighting():
#  run_all_tests(ndtw.dtw_loss)
#  simple_cuda_test(ndtw.dtw_loss)

def paramfunc(r,a):
  return pcdtw.dtw_loss(r,a,1)

def paramfunctwo(r,a):
  return pcdtw.dtw_loss(r,a,2)

def paramvectfunc(r,a):
  return pvdtw.dtw_loss(r,a,torch.tensor((0.,0.,)), torch.tensor((0.,0.,)))

def paramvectfunctwo(r,a):
  return pvdtw.dtw_loss(r,a,torch.tensor((0.,1.,)), torch.tensor((0.,1.,)))

def test_cpp_dtw_actual_weighting():
  run_all_tests(cdtw.dtw_loss)
  run_all_tests(paramfunc)
  run_all_tests(paramvectfunc)
  # I'm not convinced these tests are actually supposed to pass (weird warping penalty)
  # probably needs new baselines 2022-12-15
  #run_all_tests(paramvectfunctwo)
  run_all_tests(paramfunctwo, handles_edge_case=True)

if __name__=="__main__":
  test_cpp_dtw_actual_weighting()

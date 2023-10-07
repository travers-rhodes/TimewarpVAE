import torch
import torch.nn as nn
import cpp_dtw
import pvect_cpp_dtw  
import cpp_dtw_cuda_split
import cpp_linear_dtw


# dummy implementation of DTW to warp first to second
class IdentityVectorTimewarper(nn.Module):
  def __init__(self, **kwargs):
    super(IdentityVectorTimewarper, self).__init__()
    pass

  def timewarp_first_to_second(self, first_trajectory, second_trajectory):
    if second_trajectory is not None and not first_trajectory.shape[1] >= second_trajectory.shape[1]:
      raise Exception(f"Currenty, dummy vector timewarper only shortens trajectories. The input shapes were "+
          f"{first_trajectory.shape} and {second_trajectory.shape}")
    return first_trajectory[:,:second_trajectory.shape[1],:]
  
  def timewarp_first_and_second(self, first_trajectory, second_trajectory):
    return first_trajectory[:,:second_trajectory.shape[1],:], second_trajectory

# cpp implementation of DTW to warp first to second
class DTWVectorTimewarper(nn.Module):
  def __init__(self, 
               first_duplication_cost = None, 
               second_duplication_cost = None,
               **kwargs):
    super(DTWVectorTimewarper, self).__init__()
    if ((first_duplication_cost is None and second_duplication_cost is not None) or
        (first_duplication_cost is not None and second_duplication_cost is None)):
      raise Exception("first_duplication_cost and second_duplication_cost must be both provided or both not provided. you can't just provide one")
    self.first_duplication_cost = torch.tensor(first_duplication_cost,dtype=torch.float) if first_duplication_cost is not None else None
    self.second_duplication_cost = torch.tensor(second_duplication_cost,dtype=torch.float) if second_duplication_cost is not None else None

  def timewarp_first_to_second(self, first_trajectory, second_trajectory):
    if first_trajectory.device != torch.device("cpu"):
      if self.first_duplication_cost is not None:
        raise Exception("We currently don't support specifying the additive regularization for cuda")
      return cpp_dtw_cuda_split.dtw_warp_first_to_second(first_trajectory, second_trajectory)

    if self.first_duplication_cost is not None:
      return pvect_cpp_dtw.dtw_warp_first_to_second(first_trajectory, second_trajectory,
                self.first_duplication_cost, self.second_duplication_cost)
    else: 
      return cpp_dtw.dtw_warp_first_to_second(first_trajectory, second_trajectory)
  
  def timewarp_first_and_second(self, first_trajectory, second_trajectory):
    if first_trajectory.device != torch.device("cpu"):
      if self.first_duplication_cost is not None:
        raise Exception("We currently don't support specifying the additive regularization for cuda")
      return cpp_dtw_cuda_split.dtw_warp_first_and_second(first_trajectory, second_trajectory)
    if self.first_duplication_cost is not None:
      return pvect_cpp_dtw.dtw_warp_first_and_second(first_trajectory, second_trajectory,
                self.first_duplication_cost, self.second_duplication_cost)
    else: 
      return cpp_dtw.dtw_warp_first_and_second(first_trajectory, second_trajectory)

# cpp implementation of DTW to warp first to second
class LinearDTWVectorTimewarper(nn.Module):
  def __init__(self, vector_timewarper_lambda_regularization, **kwargs):
    super(LinearDTWVectorTimewarper, self).__init__()
    self.vector_timewarper_lambda_regularization = vector_timewarper_lambda_regularization
    pass

  def timewarp_first_to_second(self, first_trajectory, second_trajectory):
    if first_trajectory.device != torch.device("cpu"):
      raise Exception("Please copy over the cuda implementation. Currently we just support cpu")
    return cpp_linear_dtw.dtw_warp_first_to_second(first_trajectory, second_trajectory, self.vector_timewarper_lambda_regularization)
  
  def timewarp_first_and_second(self, first_trajectory, second_trajectory):
    raise Exception("Haven't implemented yet")

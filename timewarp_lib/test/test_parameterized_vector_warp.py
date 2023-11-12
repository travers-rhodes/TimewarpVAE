import torch
import unittest

import torch.nn
import timewarp_lib.parameterized_vector_time_warper as pvtw


class TestVAELearner(unittest.TestCase):
  def test_simplest_warping(self):
    # Test that the simplest warp works
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).float().reshape(1,-1,1)
    T = x.shape[1]-1
    gammadot = torch.ones(T).reshape(1,-1)/T

    warped_x = pvtw.warp(x, gammadot)

    print(warped_x, x)
    self.assertTrue(torch.allclose(warped_x, x))
  
  def test_step_warping(self):
    # Test that the simplest warp works
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).float().reshape(1,-1,1).repeat(2,1,2)
    x[:,:,1] = x[:,:,0] + 100
    T = x.shape[1]-1
    gammadot1 = torch.ones(T)/T
    gammadot2 = torch.zeros(T)
    gammadot2[:5] = 0.2
    gammadot = torch.cat([gammadot1, gammadot2]).reshape(2,T)

    warped_x = pvtw.warp(x, gammadot)
    expected_warp = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).float().reshape(1,-1,1).repeat(2,1,2)
    expected_warp[0,:,1] = expected_warp[0,:,0] + 100
    expected_warp[1,5:,:] = 10
    expected_warp[1,:5,:] = torch.arange(1,10,9/5.).unsqueeze(1)
    expected_warp[1,:,1] = expected_warp[1,:,0] + 100
    self.assertTrue(torch.allclose(warped_x, expected_warp))

if __name__ == '__main__':
  unittest.main()



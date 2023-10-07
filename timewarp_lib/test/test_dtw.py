import torch
import unittest
import numpy as np
import cpp_dtw as dtw
import cpp_dtw_cuda as dtwc
import cpp_dtw_cuda_split as dtwcsplit

class TestDTW(unittest.TestCase):
  def test_warp_first_to_second(self):
    first = torch.Tensor(np.array((0,0,0,0,0,1,2,3,2,1,0)).reshape(1,-1,1))+3
    second = torch.Tensor(np.array((0,1,2,3,2,1,0,0,0,0)).reshape(1,-1,1))+3
    newfirst = dtw.dtw_warp_first_to_second(first,second)
    np.testing.assert_almost_equal(newfirst.detach().cpu().numpy(),second.detach().cpu().numpy())
  
  def test_warp_first_to_second_cuda(self):
    device="cuda"
    first = torch.Tensor(np.array((0,0,0,0,0,1,2,3,2,1,0)).reshape(1,-1,1)).to(device)+3
    second = torch.Tensor(np.array((0,1,2,3,2,1,0,0,0,0)).reshape(1,-1,1)).to(device)+3
    newfirst = dtwc.dtw_warp_first_to_second(first,second)
    np.testing.assert_almost_equal(newfirst.detach().cpu().numpy(),second.detach().cpu().numpy())
    newfirst = dtwcsplit.dtw_warp_first_to_second(first,second)
    np.testing.assert_almost_equal(newfirst.detach().cpu().numpy(),second.detach().cpu().numpy())


if __name__ == '__main__':
  unittest.main()


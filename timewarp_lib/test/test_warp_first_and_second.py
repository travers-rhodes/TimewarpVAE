import torch
import unittest
import numpy as np
import cpp_dtw as dtw

TEST_CUDA=torch.cuda.is_available()

if TEST_CUDA:
  import cpp_dtw_cuda_split as dtw_cuda_split

class TestDTW(unittest.TestCase):
  def test_warp_first_and_second(self):
    first = torch.Tensor(np.array((0,0,0,0,0,1,2,3,2,1,0)).reshape(1,-1,1))+3
    second = torch.Tensor(np.array((0,1,2,3,2,2,1,0,0,0,0)).reshape(1,-1,1))+3
    newfirst, newsecond = dtw.dtw_warp_first_and_second(first,second)
    # remember the sqrt logic!
    fact = np.sqrt(0.2)
    expectedResultVals = ((np.array((0,0,0,0,0,1,2,3,2,2,1,0,0,0,0)) + 3 )
                      * np.array((fact,fact,fact,fact,fact,1,1,1,1,1,1,1,1,1,1)))
    expectedResult = np.zeros(23).reshape(1,-1,1)
    expectedResult[0,-15:,0] = expectedResultVals
    print(expectedResult,newfirst,newsecond)
    np.testing.assert_almost_equal(newfirst.detach().cpu().numpy(),expectedResult)
    np.testing.assert_almost_equal(newsecond.detach().cpu().numpy(),expectedResult)
  
  def test_warp_first_and_second_switched(self):
    # should give DIFFERENT answer if we role-reverse, because (square of) scale
    # should sum to 1 for each thing matched to ACTUAL.
    # So, we have a different weighting, now, of:
    actual = torch.Tensor(np.array((0,0,0,0,0,1,2,3,2,1,0)).reshape(1,-1,1))+3
    recon = torch.Tensor(np.array((0,1,2,3,2,2,1,0,0,0,0)).reshape(1,-1,1))+3
    newfirst, newsecond = dtw.dtw_warp_first_and_second(recon,actual)
    # remember the sqrt logic!
    expectedResultVals = ((np.array((0,0,0,0,0,1,2,3,2,2,1,0,0,0,0)) + 3 )
                      * np.array((1,1,1,1,1, 1,1,1, 1/np.sqrt(2),1/np.sqrt(2), 1, 1/np.sqrt(4.),1/np.sqrt(4.),1/np.sqrt(4.),1/np.sqrt(4.))))
    expectedResult = np.zeros(23).reshape(1,-1,1)
    expectedResult[0,-15:,0] = expectedResultVals
    print(expectedResult,newfirst,newsecond)
    np.testing.assert_almost_equal(newfirst.detach().cpu().numpy(),expectedResult)
    np.testing.assert_almost_equal(newsecond.detach().cpu().numpy(),expectedResult)
  
  def test_warp_first_and_second_cuda(self):
    if not TEST_CUDA:
      return
    first = torch.Tensor(np.array((0,0,0,0,0,1,2,3,2,1,0)).reshape(1,-1,1))+3
    second = torch.Tensor(np.array((0,1,2,3,2,2,1,0,0,0,0)).reshape(1,-1,1))+3
    device = "cuda"
    first = first.to(device)
    second = second.to(device)
    newfirst, newsecond = dtw_cuda_split.dtw_warp_first_and_second(first,second)
    # remember the sqrt logic!
    fact = np.sqrt(0.2)
    expectedResultVals = ((np.array((0,0,0,0,0,1,2,3,2,2,1,0,0,0,0)) + 3 )
                      * np.array((fact,fact,fact,fact,fact,1,1,1,1,1,1,1,1,1,1)))
    expectedResult = np.zeros(23).reshape(1,-1,1)
    expectedResult[0,-15:,0] = expectedResultVals
    print(expectedResult,newfirst,newsecond)
    np.testing.assert_almost_equal(newfirst.detach().cpu().numpy(),expectedResult)
    np.testing.assert_almost_equal(newsecond.detach().cpu().numpy(),expectedResult)
  

if __name__ == '__main__':
  unittest.main()

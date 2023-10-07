import unittest
import torch
import numpy as np
import timewarp_lib.utils.parametric_monotonic_function as pmf

class TestMonotonicTransformer(unittest.TestCase):
  def test_inverse(self):
    granularity = 2
    dtype=torch.float
    applier = pmf.ParameterizedMonotonicApplier(granularity, dtype=dtype)
    coeffs = torch.tensor(np.arange(granularity).reshape(-1,1), dtype=dtype)
    coeffs = torch.nn.LogSoftmax(dim=0)(coeffs) + np.log(granularity)
    ts = torch.tensor(np.array((0,0.25,0.5,0.75,1)).reshape(-1,1), dtype=dtype)
    transformed_ts = applier.apply_monotonic_transformation(coeffs, ts)
    original_expected = np.array((0,0.25,0.5,0.5+0.25*np.exp(1), 0.5+0.5*np.exp(1))).reshape(-1,1)
    new_expected = original_expected / original_expected[-1,-1]
    np.testing.assert_almost_equal(transformed_ts, new_expected)
    roundtrip_ts = applier.apply_inverse_monotonic_transformation(coeffs, transformed_ts)
    np.testing.assert_almost_equal(roundtrip_ts.numpy(), ts)
  
  def test_batch(self):
    granularity = 2
    dtype=torch.float
    applier = pmf.ParameterizedMonotonicApplier(granularity, dtype=dtype)
    coeffs = torch.tensor(np.concatenate((np.arange(granularity).reshape(1,-1),
                                          np.arange(granularity).reshape(1,-1)*2), axis=0),
                          dtype=dtype)
    coeffs = torch.nn.LogSoftmax(dim=1)(coeffs) + np.log(granularity)
    ts = torch.tensor(np.array((0,0.25,0.5,0.75,1)).reshape(1,-1,1), dtype=dtype).expand(2,-1,-1)
    transformed_ts = applier.batch_apply_monotonic_transformation(coeffs, ts)
    expected1 = np.array((0,0.25,0.5,0.5+0.25*np.exp(1), 0.5+0.5*np.exp(1))).reshape(-1,1)
    new_expected1 = expected1 / expected1[-1,:]
    print(transformed_ts)
    print(expected1)
    np.testing.assert_almost_equal(transformed_ts[0],new_expected1)
    expected2 = np.array((0,0.25,0.5,0.5+0.25*np.exp(2), 0.5+0.5*np.exp(2))).reshape(-1,1)
    new_expected2 = expected2 / expected2[-1,-1]
    np.testing.assert_almost_equal(transformed_ts[1],new_expected2)


if __name__ == '__main__':
  unittest.main()


import unittest
import torch
import numpy as np
import timewarp_lib.decoders as dt
import timewarp_lib.train_utils as tu

def get_xs_ts(batch_size, traj_len, traj_channels, dtype=torch.float):
  xs = torch.rand((batch_size, traj_len, traj_channels),dtype=dtype)
  scaled_ts = torch.rand((batch_size, traj_len, 1),dtype=dtype)
  return xs,scaled_ts

class TestVAELearner(unittest.TestCase):
  def test_curv_func_decode_traj_dim_2(self):
    dtype = torch.float
    latent_dim = 1
    traj_channels=2
    decoder = dt.FunctionStyleDecoder(latent_dim,
                                        traj_channels=traj_channels,
                                        dec_template_motion_hidden_layers=[2],
                                        dtype=dtype,
                                 useSoftmax=True)
    weight1 = np.array([[1., 0],[0,1]])
    bias1 = np.array([0,0])
    weight2 = np.array([[1., 0],[0,1]])
    bias2 = np.array([0,0])
    with torch.no_grad():
        decoder.motion_model.all_layers[0].weight.copy_(torch.tensor(weight1, dtype=dtype))
        decoder.motion_model.all_layers[0].bias.copy_(torch.tensor(bias1, dtype=dtype))
        decoder.motion_model.all_layers[1].weight.copy_(torch.tensor(weight2, dtype=dtype))
        decoder.motion_model.all_layers[1].bias.copy_(torch.tensor(bias2, dtype=dtype))

    nsamps = 6
    test_traj_len = 1
    zvalues = torch.tensor([-1,-1,0,0,1,1],dtype=dtype).reshape(nsamps, latent_dim)
    tvalues = torch.tensor([0,0,0,0,0,0],dtype=dtype).reshape(nsamps, test_traj_len, 1)
    actual = tu.curvature_loss_function(decoder.decode, zvalues, tvalues, epsilon_scale = 1, epsilon_div_zero_fix = 1e-7)
    np.testing.assert_almost_equal(4./3, actual.detach().cpu().numpy(), decimal=6)

  def test_sample_points(self):
    dtype = torch.float
    latent_dim = 1
    nsamps = 3
    test_traj_len = 3
    nnewsamps = int(test_traj_len * 3)
    zvalues = torch.tensor([-1,0,1],dtype=dtype).reshape(nsamps, latent_dim)
    tvalues = torch.tensor([[0,1,2],[0,1,2],[0,1,2]],dtype=dtype).reshape(nsamps, test_traj_len, 1)
    print(tvalues)
    zs, ts = tu.sample_latent_points_and_times(zvalues, tvalues,nnewsamps)
    expected_t_values = tvalues.transpose(0,1).reshape(9,1,1) # the output here is 0,0,0,1,1,1,2,2,2 
    np.testing.assert_almost_equal(ts.detach().cpu().numpy(), 
                                   expected_t_values.detach().cpu().numpy())
    np.testing.assert_almost_equal(zs.detach().cpu().numpy().shape, 
                                  (9,1))

if __name__ == '__main__':
  unittest.main()


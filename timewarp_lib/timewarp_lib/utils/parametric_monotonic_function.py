import torch
import torch.nn as nn
import numpy as np

class ParameterizedMonotonicApplier(nn.Module):
    def __init__(self, granularity = 10,dtype=torch.float):
        super(ParameterizedMonotonicApplier, self).__init__()
        self.granularity = granularity
        knotmatrix = np.linspace(0,1,granularity,endpoint=False).reshape(1,self.granularity)
        broadcaster = torch.tensor(np.ones(shape=(1,self.granularity)),dtype=dtype)
        knots = torch.tensor(knotmatrix, dtype=dtype)

        #https://discuss.pytorch.org/t/track-non-learnable-parameters-in-pytorch/106066/2
        self.register_buffer("knots", knots)
        self.register_buffer("broadcaster", broadcaster)
        #self.device="cpu"

    # commented out, since I think I fixed this with register_buffer above
    # be able to send to "cuda" using a simple obj = obj.to("cuda") syntax
    #def to(self, device):
    #    print(f"Sending knots to {device}")
    #    self.broadcaster = self.broadcaster.to(device)
    #    self.knots = self.knots.to(device)
    #    self.device=device
    #    return self
 
    # The input ts is of shape batchsize x timesteps x 1 
    # The input transform_coeffs is of shape batchsize x granularity
    # The output (scaled_ts) is of shape batchsize x timesteps x 1
    # and corresponds to the canonical timestamps (ie: the timesteps in the canonical
    # trajectory)
    def batch_apply_monotonic_transformation(self, transform_coeffs, ts):
        if transform_coeffs.shape[1] != self.granularity:
            raise Exception("transform_coeffs should be of shape 'batchsize x self.granularity'")
        if len(ts.shape)!=3 or ts.shape[2] != 1:
            raise Exception("ts should be of shape 'batchsize' x 'timesteps' x '1'")
        if ts.shape[0] != transform_coeffs.shape[0]:
            raise Exception("the batch size of ts and transform_coeffs should be equal")
        # broadcast ts (a batch X timesteps X 1 matrix) to become a batch X timesteps X numKnots matrix
        broadts = ts.expand([-1,-1,self.granularity])
        # tknotcoeff is a matrix giving how far each time (row) is to the right of its associated knot (column)
        tknotcoeff = broadts - self.knots.unsqueeze(0)
        # clamp to small positive range---very similar shape to HardSigmoid
        tknotclamped = tknotcoeff.clamp(0,1./self.granularity)# - 0.5/self.granularity
        # finally, multiplying each of these by a positive number and taking
        # the sum gives our monotonic function
        scaled_ts = torch.einsum("btg,bg->bt", tknotclamped, torch.exp(transform_coeffs))
        return scaled_ts.unsqueeze(2)

    def apply_monotonic_transformation(self, transform_coeffs, ts):
        if transform_coeffs.shape[0] != self.granularity or transform_coeffs.shape[1] != 1:
            raise Exception("transform_coeffs should be of shape 'self.granularity' x '1'")
        if len(ts.shape)!=2 or ts.shape[1] != 1 :
            raise Exception("ts should be of shape 'batch_to_compute' x '1'")
        # broadcast ts (a batch X 1 matrix) to become a batch X numKnots matrix
        broadts = torch.matmul(ts,self.broadcaster)
        # tknotcoeff is a matrix giving how far each ts (row) is to the right of its associated knot (column)
        tknotcoeff = broadts - self.knots
        # clamp to small positive range---very similar shape to HardSigmoid
        # finally, multiplying each of these by a positive number and taking
        # the sum gives our monotonic function
        tknotclamped = tknotcoeff.clamp(0,1./self.granularity)# - 0.5/self.granularity
        output = torch.matmul(tknotclamped,torch.exp(transform_coeffs))
        return output

    # Start with a simple, if inefficient, calculation
    def apply_inverse_monotonic_transformation(self, transform_coeffs, ts):
        if ts.shape[1] != 1:
            raise Exception("ts should be a column vector")
        if transform_coeffs.shape[0] != self.granularity or transform_coeffs.shape[1] != 1:
            raise Exception("transform_coeffs should be of shape 'self.granularity' x '1'")

        knot_step_sizes = torch.exp(transform_coeffs) * 1./self.granularity
        knot_thresholds = torch.cumsum(knot_step_sizes,axis=0).reshape(1,-1)
        how_much_past_desired_value = knot_thresholds - ts
        still_below_desired_value = how_much_past_desired_value < 0
        first_knot_past_desired_value = torch.sum(still_below_desired_value,axis=1).detach().cpu().numpy()
        inv_scaled_ts = []
        # rowind corresponds to which ts value we're computing
        # colind corresponds to which knot value we're interested in
        for rowind, colind in enumerate(first_knot_past_desired_value):
            if ts[rowind] < 0:
                inverse_value = -np.inf
            elif colind >= transform_coeffs.shape[0]:
                inverse_value = np.inf
            else:
                knot_size = knot_step_sizes[colind]
                overshot_amount = how_much_past_desired_value[rowind, colind]
                fractional_increment = 1 - overshot_amount/knot_size
                inverse_value = (colind + fractional_increment) * 1./self.granularity
            inv_scaled_ts.append(inverse_value)
        inv_scaled_ts = torch.tensor(inv_scaled_ts).reshape(-1,1).to(self.knots.device)
        return inv_scaled_ts

# timewarp_lib
Library for training VAEs on trajectory data while explicitly accounting for timewarping to align the trajectories


# Code Overview
## `scalar_timewarpers.py`
File containing objects that can apply a scalar timewarping function. The scalar timewarping function warps scalars $t \to t'$.

### API:

Application Function:
#### timewarp
```
def timewarp(self, xs, ts):
```
With `xs` the pose data and ts the scalar time indices to warp.
the shapes should be (data_dim is usually 7 for pose data):
```
xs: (batch_size, traj_len, data_dim)
ts: (batch_size, 1)
```
The output is the scaled ts:
```
scaled_ts: (batch_size, 1)
```

### Options:
* IdentityScalarTimewarper ($t'$ equals $t$)
* ModeledParameterScalarTimewarper ($t'$ is computed from a stepwise linear function off of $t$ with slopes modeled from $xs$ using a `OneDConvEmbed` embedding network)

## `encoders.py`
File containing objects that can apply an encoding function. The encoding function maps trajectories to latent values.

### API:
Application Function:
#### encode
```
def encode(self,x,scaled_ts=None):
```
A function to take in an input trajectory (and optionally, the time indices associated with points in the trajectory) and output the encoded latent value (and encoding distributional values).
The input shapes should be:
```
x: (batch_size, traj_len, traj_channels)
scaled_ts: (batch_size, traj_len, 1)
```
Output:
```
mu: an embedding of the whole trajectory in the shape (batch_size, latent_dim)
logvar: an associated logvar for noise associated with the embedding also in the shape (batch_size, latent_dim)
```
### Options:
* OneDConvEncoder (1D convolutions followed by fully-connected layers. Does not use `scaled_ts` parameter)
* SelfAttentionTransformerEncoder (Self-attention based architecture. Can optionally use `scaled_ts` parameter)
* TransformerEncoder (Attention-based architecture. Can optionally use `scaled_ts` parameter)

## `decoders.py`
File containing objects that can apply an encoding function. The encoding function maps latent values to trajectories.

### API:
Application Function:
#### decode
```
def decode(self, zs, ts):
```
A function to take in the latent value `zs` and the time indices associated with (desired) points in the output trajectory `ts` and outputs a trajectory.
The input shapes should be:
```
zs: (batchsize, latent_dim)
ts: (batchsize, traj_len, 1)
```
Output:
```
trajectory: (batch_size, traj_len, traj_channels)
```

### Options:
* ComplicatedFunctionStyleDecoder: (Two modules of fully connected layers, one which just takes in timestep (generating a vector) and one which just takes in latent value (generating a matrix). The results of these are then multiplied together using matrix multiplication to give the output pose for that timestep for that latent value. This is repeated for all the different timesteps to generate the full trajectory)
* FunctionStyleDecoder: (fully connected layers which repeatedly takes in the timestep concatenated with the latent value and outputs pose for each timesteps)
* OneDConvDecoderUpsampling: (fully connected followed by nn.Conv1d layers without striding. Instead of striding, before each convolution we (can) repeat elements to increase the length of the trajectory, using the `dec_gen_upsampling_factors` parameter.  Does not use `ts` parameter)
* OneDConvDecoder: (fully connected followed by nn.ConvTranspose1d layers (upconvolution). Does not use `ts` parameter)

## `vector_timewarpers.py`
File containing objects that can apply timewarping to a trajectory. The timewarping maps the first trajectory to best align with the second trajectory.

### API:
Application Function:
#### timewarp_first_to_second
```
def timewarp_first_to_second(self, first_trajectory, second_trajectory):
```
Create a new trajectory whose poses are computed from the `first_trajectory`, but the timing at which those poses are reached are chosen to make the resulting trajectory similar to `second_trajectory`. Average poses if multiple poses in `first_trajectory` correspond to the same pose in `second_trajectory`.

Input dimensions:
```
first_trajectory: (batch_size, traj_len_first, traj_channels)
second_trajectory: (batch_size, traj_len_second, traj_channels)
```
Output:
```
trajectory: (batch_size, traj_len_second, traj_channels)
```
#### timewarp_first_and_second
```
def timewarp_first_and_second(self, first_trajectory, second_trajectory):
```
Create two new trajectories, one whose poses are computed from the `first_trajectory`, 
and one whose poses are computed from the `second_trajectory`.
The timing of both trajectories are chosen by repeating different timesteps in the trajectories so that matching timesteps have similar poses.
The poses within both output trajectories are scaled by dividing by the number times the
same timestamp of `second_trajectory` was repeated. Equivalently, that is the number of different
timesteps in `first_trajectory` that are matched to this timestamp in the `second_trajectory`.
By scaling in this way, the mean squared error of the two output trajectories 
is equal to the following calculation:
Match each pose in `second_trajectory` to all its associated poses in `first_trajectory`.
For each pose in `second_trajectory`, compute the mean square difference between it and its matched poses in the `first_trajectory`.
Finally, average over all those errors (one for each timestamp in `second_trajectory`).

Input dimensions:
```
first_trajectory: (batch_size, traj_len_first, traj_channels)
second_trajectory: (batch_size, traj_len_second, traj_channels)
```
Output:
```
warped_first_trajectory: (batch_size, extended_traj_len, traj_channels)
warped_second_trajectory: (batch_size, extended_traj_len, traj_channels)
```
### Options:
* IdentityVectorTimewarper: Truncate or repeat last element of first trajectory to match second trajectory (if same lengths, do nothing)
* DTWVectorTimewarper: Use standard tabular DTW algorithm to match timesteps between trajectories. 

## `vae_template.py`
Template file to combine a scalar_timewarper, encoder, and decoder to create a Variational Auto-Encoder model.

### API:
Application Function:

#### noiseless_forward
```
def noiseless_forward(self, xs, ts, training_index=None):
```
Timewarp the `ts` to `scaled_ts`, encode the `xs` (and `scaled_ts`) to a latent value, and then decode the latent value (with the `scaled_ts`) back to a trajectory.
Input shapes:
```
xs: (batch_size, traj_len, traj_dim)
ts: (batch_size, traj_len, 1)
training_index: (1) [not really supported...]
```
Output shapes:
```
x: (batch_size, traj_len, traj_channels)
scaled_ts: (batch_size, traj_len, traj_channels)
```
#### forward
```
def forward(self, xs, ts, training_index = None):
```
Timewarp the `ts` to `scaled_ts`, encode the `xs` (and `scaled_ts`) to a latent value, add latent noise, and then decode the noisy latent value (with the `scaled_ts`) back to a trajectory.


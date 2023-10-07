import torch
import numpy as np
import numba

TESTING = True

# do a kind of simulation of brownian motion to generate a nice, noisy
# monotonically increasing time-warping function from 0 to 1
def get_warped_xvals(noise,numfirsts,numts):
  _, newxs = lininterp(get_first_warped(noise,numfirsts).reshape(-1,1),get_first_warped(noise,numfirsts).reshape(-1,1),numts)
  return newxs


# do a kind of simulation of brownian motion to generate a nice, noisy 
# monotonically increasing time-warping function from 0 to 1.
# in order ot make the noise nicely smooth, set num_knots to be small
# (like, say, 10), and then lininterp to the desired numts later
def get_first_warped(noise,num_knots):
    xnoise_steps = (np.random.uniform(size=num_knots-1))**2 * noise
    xnoise = np.concatenate(([0],np.cumsum(xnoise_steps)))
    xdata = (np.linspace(0,1,num_knots) + xnoise)/(1 + xnoise[-1])
    return xdata

def add_timing_noise(data, training_data_timing_noise):
  if training_data_timing_noise == 0:
    return data

  np_yvals = data.detach().cpu().numpy()
  num_trajs, num_ts, _ = np_yvals.shape
  new_yvals = []
  numfirsts = 10
  for i in range(num_trajs):
    warped_xvals = get_warped_xvals(training_data_timing_noise, numfirsts, num_ts)
    _, new_yval = lininterp(warped_xvals, np_yvals[i], num_ts)
    new_yvals.append(new_yval)
  new_yvals = np.array(new_yvals)
  newdata = torch.tensor(new_yvals, dtype = data.dtype, device = data.device)
  return newdata


# given xvalue/yvalue pairs, resample them uniformly along new x-values
# linearly interpolating the yvalues at the new, uniform xvalues.
# Calculations are all done with numpy arrays here
@numba.njit
def lininterp(xvals, yvals, numsamples):
    newxvals = np.linspace(np.min(xvals), np.max(xvals), numsamples).astype(np.float32)
    newyvals = np.empty(shape=(numsamples,yvals.shape[1]),dtype=np.float32)
    curxind = 0
    for i, newxval in enumerate(newxvals):
        while curxind < len(xvals) - 1 and xvals[curxind+1] < newxval:
            curxind += 1
        if curxind == len(xvals) - 1:
            newyval = yvals[curxind]
        else:
            ydiff = yvals[curxind + 1] - yvals[curxind]
            xdiff = xvals[curxind + 1] - xvals[curxind]
            if xdiff == 0:
                newyval = yvals[curxind]
            else:
                curxdiff = newxval - xvals[curxind]
                newyval = curxdiff/xdiff * ydiff + yvals[curxind]
        newyvals[i,:] = newyval
    return newxvals, newyvals


# given a noiseless encoding and a (diagonal gaussian) noise around each encoding
# compute the KL divergence for the given batch.
def kl_loss_term(mu, logvar):
    batch_size = mu.shape[0]
    if TESTING:
        assert logvar.shape[0] == batch_size, "logvar should have batch_size as first dimension"

    ########### mu is of shape batchsize x latent_dim
    mu_error = -0.5 * torch.sum(- mu.pow(2))/batch_size
    if TESTING:
        assert len(mu_error.shape)==0,"mu_error should be scalar"

    ########### mu is of shape batchsize x latent_dim
    # note that the 1 is getting broadcast batchsize x latent_dim times 
    # (so this calc correctly implements Appendix B of Auto-Encoding Variational Bayes)
    logvar_error = -0.5 * torch.sum(1 + logvar - logvar.exp())/batch_size
    if TESTING:
        assert len(logvar_error.shape)==0,"logvar_error should be scalar"

    #print("kl content", latent_dim + logvar - mu.pow(2) - logvar.exp())
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = logvar_error + mu_error

    return KLD

# sample some new points, based on the distribution of input zs and ts
# compute and return the jlone loss of the decode_func at each point
# oversampling scale is because we use a multivariate gaussian to model the data
# and since the data is not actually gaussian distributed 
# (eg: time is uniform sampled then scaled)
# we expand the sampling multivariate standard deviation by oversampling_scale
# to ensure full (excess) coverage
##
## Input:
# decode_func(zs : tensor(batchsize,latentdim), ts(batchsize,1)) -> tensor(batchsize, model_output_shape)
def jlone_loss_term(decode_func, noisy_zs, scaled_ts, num_new_samp_points, epsilon_scale):
  loss_zs, loss_ts = sample_latent_points_and_times(noisy_zs, scaled_ts, num_new_samp_points)
  jlone_loss = jacobian_lone_loss_function(decode_func, loss_zs, loss_ts, epsilon_scale)
  return jlone_loss

def curvature_loss_term(decode_func, noisy_zs, scaled_ts, num_new_samp_points, epsilon_scale, epsilon_div_zero_fix):
  loss_zs, loss_ts = sample_latent_points_and_times(noisy_zs, scaled_ts, num_new_samp_points)
  curvature_loss = curvature_loss_function(decode_func, loss_zs, loss_ts, epsilon_scale, epsilon_div_zero_fix)
  return curvature_loss

def second_deriv_loss_term(decode_func, noisy_zs, scaled_ts, num_new_samp_points, epsilon_scale):
  loss_zs, loss_ts = sample_latent_points_and_times(noisy_zs, scaled_ts, num_new_samp_points)
  curvature_loss = second_deriv_loss_function(decode_func, loss_zs, loss_ts, epsilon_scale)
  return curvature_loss

def curvature_loss_function(decode_func, zvalues, tvalues, epsilon_scale, epsilon_div_zero_fix):
    assert tvalues.shape[1] == 1 and tvalues.shape[2] == 1, "only apply this to trajectories with single timesteps for now"
    assert zvalues.shape[0] == tvalues.shape[0], "tensor params should have same batch size" 
    assert len(zvalues.shape) == 2, "latent dim param should be two dimensional" 
    device=zvalues.device
    # 2021-12-15: turns out that torch.normal can return exactly zero. fun fact.
    # resampling is the easy/inefficient fix
    computed_valid_hvalues = False
    while not computed_valid_hvalues:
        hvalues_raw = torch.normal(torch.zeros(size=zvalues.shape, device=device), 
                                   torch.ones(size=zvalues.shape, device=device))   
        hvalues_length = torch.sqrt(torch.sum(torch.square(hvalues_raw), axis=1)).reshape(-1,1)  
        hvalues = hvalues_raw/hvalues_length * epsilon_scale
        computed_valid_hvalues = not torch.any(hvalues.isnan())
    zvalues_detached = torch.tensor(zvalues.detach().cpu().numpy(), dtype=torch.float32).to(device)
    z_plus_hvalues_detached = torch.tensor((zvalues+hvalues).detach().cpu().numpy(), dtype=torch.float32).to(device)
    z_minus_hvalues_detached = torch.tensor((zvalues-hvalues).detach().cpu().numpy(), dtype=torch.float32).to(device)
    modeled_z = decode_func(zvalues_detached, tvalues)
    modeled_z_plus_h = decode_func(z_plus_hvalues_detached, tvalues)
    modeled_z_minus_h = decode_func(z_minus_hvalues_detached, tvalues)

    # we want low curvature of poses as func of latent, not trajectories as func of latent.
    # luckily (for now) we're only implementing on func_vae
    # for which we should only be passing in a trajectory with a single timestep
    # so we should just have one pose here

    assert modeled_z.shape[1] == 1, "only apply this to funcvae so that the modeled output also only has single timesteps for now"

    forward_length = detached_length(modeled_z_plus_h - modeled_z) + epsilon_div_zero_fix
    backward_length = detached_length(modeled_z - modeled_z_minus_h) + epsilon_div_zero_fix
    curv_est = ((modeled_z_plus_h - modeled_z)/forward_length - (modeled_z - modeled_z_minus_h)/backward_length)/((forward_length + backward_length)/2)
    loss = torch.sum(torch.square(curv_est))/curv_est.shape[0]
    return loss

def detached_length(vector):
    all_dims_but_first = list(range(len(vector.shape)))[1:]
    new_shape = np.ones(len(vector.shape))
    new_shape[0] = -1
    new_shape = list(new_shape.astype(int))
    length = torch.sqrt(torch.sum(torch.square(vector), dim=all_dims_but_first)).detach().reshape(new_shape)
    return length

def second_deriv_loss_function(decode_func, zvalues, tvalues, epsilon_scale):
    device = zvalues.device
    # 2021-12-15: turns out that torch.normal can return exactly zero. fun fact.
    # resampling is the easy/inefficient fix
    computed_valid_hvalues = False
    while not computed_valid_hvalues:
        hvalues_raw = torch.normal(torch.zeros(size=zvalues.shape, device=device), 
                                   torch.ones(size=zvalues.shape, device=device))   
        hvalues_length = torch.sqrt(torch.sum(torch.square(hvalues_raw), axis=1)).reshape(-1,1)  
        hvalues = hvalues_raw/hvalues_length * epsilon_scale
        computed_valid_hvalues = not torch.any(hvalues.isnan())
    zvalues_detached = torch.tensor(zvalues.detach().cpu().numpy(), dtype=torch.float32).to(device)
    z_plus_hvalues_detached = torch.tensor((zvalues+hvalues).detach().cpu().numpy(), dtype=torch.float32).to(device)
    z_minus_hvalues_detached = torch.tensor((zvalues-hvalues).detach().cpu().numpy(), dtype=torch.float32).to(device)
    #print(zvalues.shape, hvalues.shape, hvalues_length.shape)    
    modeled_z = decode_func(zvalues_detached, tvalues)
    modeled_z_plus_h = decode_func(z_plus_hvalues_detached, tvalues)
    modeled_z_minus_h = decode_func(z_minus_hvalues_detached, tvalues)
    second_est = (modeled_z_plus_h + modeled_z_minus_h - 2 * modeled_z)/(epsilon_scale**2)
    error = torch.sqrt(torch.sum(torch.square(second_est))/second_est.shape[0])
    return error

def jlpointfive_loss_term(decode_func, noisy_zs, scaled_ts, num_new_samp_points, epsilon_scale):
  loss_zs, loss_ts = sample_latent_points_and_times(noisy_zs, scaled_ts, num_new_samp_points)
  jlone_loss = jacobian_lpointfive_loss_function(decode_func, loss_zs, loss_ts, epsilon_scale)
  return jlone_loss

def jacobian_lpointfive_loss_function(decode_func, zs, ts, epsilon_scale, epsilon = 1e-6):
    # jacobian shape is (latent_dim, batch_size, output_model_shape)
    # where the first batch_size rows correspond to the first latent dimension, etc.
    jacobian = compute_generator_jacobian_optimized(decode_func, zs, ts, epsilon_scale)
    latent_dim = jacobian.shape[0]
    batch_size = jacobian.shape[1]
    jacobian = jacobian.reshape((latent_dim, batch_size, int(np.prod(jacobian.shape[2:]))))
    obs_dim = jacobian.shape[2]
    loss = torch.sum(torch.sqrt(torch.abs(jacobian)+epsilon))/batch_size
    assert len(loss.shape)==0, "loss should be a scalar"
    return(loss)

def jacobian_lone_loss_function(decode_func, zs, ts, epsilon_scale):
    # jacobian shape is (latent_dim, batch_size, output_model_shape)
    # where the first batch_size rows correspond to the first latent dimension, etc.
    jacobian = compute_generator_jacobian_optimized(decode_func, zs, ts, epsilon_scale)
    latent_dim = jacobian.shape[0]
    batch_size = jacobian.shape[1]
    jacobian = jacobian.reshape((latent_dim, batch_size, int(np.prod(jacobian.shape[2:]))))
    obs_dim = jacobian.shape[2]
    loss = torch.sum(torch.abs(jacobian))/batch_size
    assert len(loss.shape)==0, "loss should be a scalar"
    return(loss)

# output shape is (latent_dim, batch_size, model_output_shape)
def compute_generator_jacobian_optimized(decode_func, zs, ts, epsilon_scale):
    batch_size = zs.shape[0]
    latent_dim = zs.shape[1]
    device = zs.device
    # repeat "tiles" like ABCABCABC (not AAABBBCCC)
    # note that we detach the embedding here, so we should hopefully
    # not be pulling our gradients further back than we intend
    encoding_rep = zs.repeat(latent_dim + 1,1).detach().clone()
    ts_rep = ts.repeat(latent_dim + 1,1).detach().clone()
    # define our own repeat to work like "AAABBBCCC"
    delta = torch.eye(latent_dim)\
                .reshape(latent_dim, 1, latent_dim)\
                .repeat(1, batch_size, 1)\
                .reshape(latent_dim*batch_size, latent_dim)
    delta = torch.cat((delta, torch.zeros(batch_size,latent_dim))).to(device)
    # we randomized this before up to epsilon_scale,
    # but for now let's simplify and just have this equal to epsilon_scale.
    # I'd be _very_ impressed if the network can figure out to make the results
    # periodic with this frequency in order to get around this gradient check.
    epsilon = epsilon_scale     
    encoding_rep += epsilon * delta
    recons = decode_func(encoding_rep, ts_rep)
    temp_calc_shape = [latent_dim+1,batch_size] + list(recons.shape[1:])
    recons = recons.reshape(temp_calc_shape)
    recons = (recons[:-1] - recons[-1])/epsilon
    return(recons)


# noisy_zs and scaled_ts should be torch tensors
# of shapes (batch_size, latent_dim)
# and (batch_size, 1) respectively
def sample_latent_points_and_times(noisy_zs, scaled_ts, num_new_samp_points):
  # The strategy here is to compute the covariance of latent values and _all_ sampled timestamps 
  # (ie: modeling the whole vector  (latent1, latent2, ... latentN, time1, time2, ...timen))
  # and then recreate new sampled latents with timestamps.
  assert noisy_zs.shape[0] == scaled_ts.shape[0], f"tensor params should have same batch size" 
  scaled_ts = scaled_ts.squeeze(2)
  assert len(noisy_zs.shape) == len(scaled_ts.shape), f"tensor params (after squeezing) should have same number of dimensions but the shapes were {noisy_zs.shape} and {scaled_ts.shape}" 
  assert len(noisy_zs.shape) == 2, "tensor params should be two dimensional (after squeezing)" 
  num_timesteps = scaled_ts.shape[1]
  assert num_new_samp_points % num_timesteps == 0, "new rule: you have to sample a multiple of the number of timesteps in the training trajectories"
  embedding_dimension = noisy_zs.shape[1]
  device = noisy_zs.device
  dtype = noisy_zs.dtype
  samp_embs = torch.hstack((noisy_zs,scaled_ts))
  samp_mean = torch.mean(samp_embs.T, axis=1).detach().cpu().numpy()
  samp_cov = np.cov(samp_embs.T.detach().cpu().numpy())
  if samp_cov.size==1:
    samp_cov = samp_cov.reshape((1,1))
  sampled_latents = []
  sampled_scaled_ts = []
  for timindex in range(num_timesteps):
    #sample both zs and ts
    samp_zs_and_ts = np.random.multivariate_normal(mean=samp_mean, cov=samp_cov,
                            size=int(num_new_samp_points/num_timesteps))
    # but only use one of the timesteps
    # (so we don't reuse the same latent value many times for different timesteps)
    # the result is MANY latents, each paired with a trajectory of only one step
    samp_zs = samp_zs_and_ts[:,:embedding_dimension]
    samp_ts = samp_zs_and_ts[:,embedding_dimension + timindex]
    sampled_latents.append(samp_zs)
    sampled_scaled_ts.append(samp_ts)
  sampled_latents = np.array(sampled_latents).reshape(num_new_samp_points, embedding_dimension)
  sampled_scaled_ts = np.array(sampled_scaled_ts).reshape(num_new_samp_points, 1, 1)
  sampled_zs = torch.tensor(sampled_latents, dtype=dtype).to(device)
  sampled_ts = torch.tensor(sampled_scaled_ts, dtype=dtype).to(device)
  return (sampled_zs, sampled_ts)

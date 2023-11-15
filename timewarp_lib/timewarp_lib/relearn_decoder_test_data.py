#! /usr/env/bin python3
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import numpy as np
import os
import wandb

import timewarp_lib.load_model as lm
import timewarp_lib.train_utils as tu

import timewarp_lib.vector_timewarpers as vtw
import timewarp_lib.parse_model_parameters as pmp

class MockedOptimizer():
  def zero_grad(self):
    pass
  def step(self):
    pass

def initialize_optimizers(hi, time_optim_lr, dec_optim_lr, emb_lr, time_eps, dec_eps, emb_eps, useAdam, decoding_l2_weight_decay, learn_decoder_variance, noise_lr, noise_eps):
  if useAdam:
    time_optim =  (torch.optim.Adam(hi.scalar_timewarper.parameters(),                lr=time_optim_lr, eps=time_eps)
        if len(list(hi.scalar_timewarper.parameters())) > 0
        else MockedOptimizer())
    dec_optim =     torch.optim.Adam([p for (n,p) in hi.decoder.named_parameters() if n != "log_decoder_variance"], lr=dec_optim_lr,   eps=dec_eps, weight_decay=decoding_l2_weight_decay)
    noise_optim =     (torch.optim.SGD([p for (n,p) in hi.decoder.named_parameters() if n == "log_decoder_variance"], lr=noise_lr)
                       if learn_decoder_variance
                        else MockedOptimizer())
    emb_optim =   MockedOptimizer() 
  else:
    time_optim =   (torch.optim.Adagrad(hi.scalar_timewarper.parameters(), lr=time_optim_lr)
        if len(hi.scalar_timewarper.parameters()) > 0
        else MockedOptimizer())
    dec_optim =     torch.optim.Adagrad([p for (n,p) in hi.decoder.named_parameters() if n != "log_decoder_variance"], lr=dec_optim_lr, weight_decay=decoding_l2_weight_decay)
    noise_optim =     (torch.optim.SGD([p for (n,p) in hi.decoder.named_parameters() if n == "log_decoder_variance"], lr=noise_lr)
                       if learn_decoder_variance
                        else MockedOptimizer())
    emb_optim =   MockedOptimizer() 
  return (time_optim, dec_optim, emb_optim, noise_optim)


def train_model(old_saved_model_dir, **kwargs):

  config = dict(**kwargs)
  
  log_to_wandb_name = (kwargs["log_to_wandb_name"] 
                             if "log_to_wandb_name" in kwargs else None)

  device = kwargs["device"]
  pre_time_learning_epochs = kwargs["pre_time_learning_epochs"]
  beta = kwargs["beta"]

  modeldatafileobj = np.load(f"{old_saved_model_dir}/saved_model_info.npz", allow_pickle=True)
  model_info = {key : (modeldatafileobj[key] if key != "initialization_function" else True) for key in modeldatafileobj.keys()}
  loaded_model = lm.LoadedModel(old_saved_model_dir,device=device)
  hi, vector_timewarper = loaded_model.model, loaded_model.vector_timewarper

  datafile = kwargs["datafile"]
  dtype = kwargs["dtype"]
  loaded_data_dict = np.load(datafile)
  ## Load the training data 
  training_data = loaded_data_dict["train"]
  ydata = torch.tensor(training_data,dtype=dtype).to(device)
  num_trajs, numts, traj_channels = ydata.shape
  
  kwargs["traj_len"] = numts 
  config["traj_len"] = numts 
  kwargs["traj_channels"] = traj_channels 
  config["traj_channels"] = traj_channels


  # overwrite all the data in model_info
  for key in config.keys():
    model_info[key] = config[key]


  model_save_dir = kwargs["model_save_dir"]
  num_epochs = kwargs["num_epochs"]
  ## Generic
  scalar_timewarping_lr = kwargs["scalar_timewarping_lr"]
  scalar_timewarping_eps = kwargs["scalar_timewarping_eps"]
  encoding_lr = kwargs["encoding_lr"]
  encoding_eps = kwargs["encoding_eps"]
  decoding_lr = kwargs["decoding_lr"]
  decoding_eps = kwargs["decoding_eps"]
  decoding_l2_weight_decay = kwargs["decoding_l2_weight_decay"]
  decoding_spatial_derivative_regularization = kwargs["decoding_spatial_derivative_regularization"]
  learn_decoder_variance = kwargs["learn_decoder_variance"] 
  noise_lr = kwargs["noise_lr"] if learn_decoder_variance else 0.
  noise_eps = kwargs["noise_eps"] if learn_decoder_variance else 0.
  useAdam = kwargs["useAdam"],
  logname = kwargs["logname"]
  if "scalar_timewarper_timereg" in kwargs:
    scalar_timewarper_timereg = kwargs["scalar_timewarper_timereg"]
  else:
    scalar_timewarper_timereg = 0
  if "scalar_timewarper_endpointreg" in kwargs:
    scalar_timewarper_endpointreg = kwargs["scalar_timewarper_endpointreg"]
  else:
    scalar_timewarper_endpointreg = 0
  # go this many epochs before learning time warping
  pre_time_learning_epochs = kwargs["pre_time_learning_epochs"]
  training_data_added_timing_noise = kwargs["training_data_added_timing_noise"]
  # how big steps to use when calculating curvature
  curv_loss_penalty_weight = kwargs["curv_loss_penalty_weight"] 
  curv_loss_epsilon_scale = (kwargs["curv_loss_epsilon_scale"] 
                             if "curv_loss_epsilon_scale" in kwargs else None)
  curv_loss_num_new_sampling_points = (kwargs["curv_loss_num_new_sampling_points"] 
                             if "curv_loss_num_new_sampling_points" in kwargs else None)
  curv_loss_divide_by_zero_epsilon = (kwargs["curv_loss_divide_by_zero_epsilon"] 
                             if "curv_loss_divide_by_zero_epsilon" in kwargs else None)
  batch_size = (int)(kwargs["batch_size"])
  if "vector_timewarper_warps_recon_and_actual" in kwargs:
    vector_timewarper_warps_recon_and_actual = kwargs["vector_timewarper_warps_recon_and_actual"]
  else:
    vector_timewarper_warps_recon_and_actual = False
  latent_dim = kwargs["latent_dim"]

  ## model_save_dir should NOT already exist
  os.makedirs(model_save_dir)

  hi.decoder = pmp.parse_decoder(**kwargs) 
  hi.scalar_timewarper = pmp.parse_scalar_timewarper(**kwargs)
  hi = hi.to(device)

  if log_to_wandb_name is not None:
    run = wandb.init(project="iclr24-project", entity="teamtravers", config=config, reinit=True, group=log_to_wandb_name)

  ### Test Data Section
  ## Load the test data 
  test_data = loaded_data_dict["test"]
  test_ydata = torch.tensor(test_data,dtype=dtype).to(device)
  num_test_trajs, num_test_ts, _ = test_ydata.shape
  test_tdata = torch.tensor(np.linspace(0,1,num_test_ts).reshape(1,-1),dtype=dtype).to(device).expand(num_test_trajs,num_test_ts).unsqueeze(2)
  torch_train_data = torch.utils.data.TensorDataset(test_tdata, test_ydata)
  test_dataloader = torch.utils.data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=True)
  test_time_identity_vector_timewarper = vtw.IdentityVectorTimewarper()
  test_time_dtw_vector_timewarper = vtw.DTWVectorTimewarper()
  ###


  tdata = torch.tensor(np.linspace(0,1,numts),dtype=dtype).to(device).expand(num_trajs,numts).unsqueeze(2)

  torch_train_data = torch.utils.data.TensorDataset(tdata, ydata)
  training_dataloader = torch.utils.data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=True)

  log_dir = logname
  if log_dir is None:
    raise Exception("Please name your logdir using the logname parameter")
  writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir) #type: ignore (this comment mutes a bug in pyright that thinks we can't access SummaryWriter)
      
  (time_optim, dec_optim, emb_optim, noise_optim) = initialize_optimizers(hi, scalar_timewarping_lr, decoding_lr, encoding_lr, scalar_timewarping_eps, decoding_eps, encoding_eps, useAdam, decoding_l2_weight_decay, learn_decoder_variance, noise_lr, noise_eps)
  num_batches_seen = 0
  record_loss_every = 100


  # each epoch, use the previous embeddings to sample from for curvvae loss (if sampling each batch)
  previous_all_embs = None
  previous_all_scaled_ts = None
  for i in range(num_epochs):
      dec_optim.zero_grad()
      if scalar_timewarping_lr > 0:
        time_optim.zero_grad()
      emb_optim.zero_grad()
      recon_loss = torch.tensor(0., dtype=dtype).to(device)
      kld_loss = torch.tensor(0.,dtype=dtype).to(device)
      spatial_derivative_loss = torch.tensor(0.,dtype=dtype).to(device)
      sum_square_emb = torch.tensor(0.,dtype=dtype).to(device)
      time_weight_loss = torch.tensor(0.,dtype=dtype).to(device)
      time_endpoint_loss = torch.tensor(0.,dtype=dtype).to(device)
      all_embs = [] 
      all_scaled_ts = []
      max_decoder_grad_norm = None
      max_encoder_grad_norm = None
      max_timewarp_grad_norm = None
      for (training_ts_for_mu, training_ys_for_mu_raw) in training_dataloader:
        dec_optim.zero_grad()
        if scalar_timewarping_lr > 0:
            time_optim.zero_grad()
        emb_optim.zero_grad()
        # 2023-08-30 we need to regularize encoder too (sadly) since we are measuring round-trip test error.
        # We add timing noise as our regularization strategy.
        training_ys_for_mu = tu.add_timing_noise(training_ys_for_mu_raw, training_data_added_timing_noise)
        # 2023-08-29 since we've added dropout, we need to pay attention to whether we're doing train or eval
        hi.train()
        estimate, meanemb, logvaremb, noisyemb, scaled_ts, noisy_emb_node = hi.forward_with_noisy_embedding_node(training_ys_for_mu, training_ts_for_mu)
        all_embs.append(noisyemb.detach())
        all_scaled_ts.append(scaled_ts.detach())
        if vector_timewarper_warps_recon_and_actual:
          timewarped_recon, timewarped_actual = vector_timewarper.timewarp_first_and_second(estimate, training_ys_for_mu)
          batch_recon_loss = nn.functional.mse_loss(timewarped_recon, timewarped_actual, reduction="sum")
          recon_loss += batch_recon_loss 
        else:
          timewarped_estimate = vector_timewarper.timewarp_first_to_second(estimate, training_ys_for_mu)
          batch_recon_loss = nn.functional.mse_loss(timewarped_estimate, training_ys_for_mu, reduction="sum")
          recon_loss += batch_recon_loss 

        batch_recon_loss = batch_recon_loss / torch.exp(hi.decoder.log_decoder_variance)
        # always just do KL loss on the spatial component (which is the last latent_dim cols)
        # which is the last latent_dim elements
        batch_kld_loss = tu.kl_loss_term(meanemb[:,-latent_dim:], logvaremb[:,-latent_dim:])*len(training_ts_for_mu)
        kld_loss += batch_kld_loss
       
        if device == "cuda":
          s = torch.cuda.Stream()

          # Safe, grads are used in the same stream context as backward()
          with torch.cuda.stream(s):
            grad_value, = torch.autograd.grad(estimate, noisy_emb_node, grad_outputs=torch.ones(estimate.shape).to(device), create_graph=True)
            spatial_derivative_loss += torch.sum(torch.square(grad_value)).detach()
        else:
          grad_value, = torch.autograd.grad(estimate, noisy_emb_node, grad_outputs=torch.ones(estimate.shape).to(device), create_graph=True)
          spatial_derivative_loss += torch.sum(torch.square(grad_value)).detach()

        sum_square_emb += torch.sum(torch.square(meanemb))
        if scalar_timewarping_lr > 0:
          scalar_timewarper_parameters = hi.scalar_timewarper.get_parameters_from_poses(training_ys_for_mu)
          # log(x) = scalar_timewarper_parameters
          # x      = torch.exp(scalar_timewarper_parameters)
          # and we want to regularize (x-1) * log(x)
          batch_time_weight_loss = torch.sum((torch.exp(scalar_timewarper_parameters)-1) * scalar_timewarper_parameters) # discourage time change
          time_weight_loss += batch_time_weight_loss
          time_endpoint_loss += torch.sum(torch.square(torch.log(torch.mean(torch.exp(scalar_timewarper_parameters), dim=1)))) # encourage all trajectories to have the same endpoint
        this_batch_size = training_ys_for_mu_raw.shape[0]
        batch_recon_loss = batch_recon_loss/(this_batch_size * numts)
        batch_recon_loss += (1/2) * latent_dim * hi.decoder.log_decoder_variance
        batch_kld_loss = batch_kld_loss/this_batch_size
        loss = (batch_recon_loss
            + beta * batch_kld_loss) 
        if scalar_timewarping_lr > 0:
          batch_time_weight_loss = batch_time_weight_loss/(this_batch_size * hi.scalar_timewarper.timewarp_parameter_encoder.latent_dim)
        # only do time_regularization if we've passed a certain amount of training
        if i >= pre_time_learning_epochs and scalar_timewarper_timereg > 0:
          loss += scalar_timewarper_timereg * batch_time_weight_loss
        # curvature loss requires a batch in order to sample a bunch of points
        # use the batch from the pevious epoch
        if curv_loss_penalty_weight != 0 and previous_all_embs is not None:# or kwargs["decoder_name"] == "functional_decoder":
          curv_loss = tu.curvature_loss_term(hi.decoder.decode, previous_all_embs, previous_all_scaled_ts, 
                            curv_loss_num_new_sampling_points, curv_loss_epsilon_scale, curv_loss_divide_by_zero_epsilon)
          loss += curv_loss_penalty_weight * curv_loss
        loss.backward()

#https:/iscuss.pytorch.org/t/check-the-norm-of-gradients/27961/8
        decoder_grads = [param.grad.detach().flatten() for param in hi.decoder.parameters() if param.grad is not None]
        decoder_norm = torch.cat(decoder_grads).norm()
        encoder_grads = [param.grad.detach().flatten() for param in hi.encoder.parameters() if param.grad is not None]
        encoder_norm = torch.cat(encoder_grads).norm()
        timewarp_grads = [param.grad.detach().flatten() for param in hi.scalar_timewarper.parameters() if param.grad is not None]
        timewarp_norm = torch.cat(timewarp_grads).norm()

        if max_decoder_grad_norm is None or max_decoder_grad_norm < decoder_norm:
          max_decoder_grad_norm = decoder_norm
        if max_encoder_grad_norm is None or max_encoder_grad_norm < encoder_norm:
          max_encoder_grad_norm = encoder_norm
        if max_timewarp_grad_norm is None or max_timewarp_grad_norm < timewarp_norm:
          max_timewarp_grad_norm = timewarp_norm

        dec_optim.step()
        if learn_decoder_variance:
          noise_optim.step()
        if scalar_timewarping_lr > 0 and i >= pre_time_learning_epochs:
          time_optim.step()
        emb_optim.step()

      all_embs_torch = torch.cat(all_embs).detach()
      all_scaled_ts_torch = torch.cat(all_scaled_ts).detach()
      previous_all_embs = all_embs_torch
      previous_all_scaled_ts = all_scaled_ts_torch

      if curv_loss_penalty_weight != 0:# or kwargs["decoder_name"] == "functional_decoder":
        curv_loss = tu.curvature_loss_term(hi.decoder.decode, all_embs_torch, all_scaled_ts_torch, 
                    curv_loss_num_new_sampling_points, curv_loss_epsilon_scale, curv_loss_divide_by_zero_epsilon)
      else:
        # fill with 0, just for logging
        curv_loss = torch.tensor(0., dtype=dtype).to(device)
      recon_loss = recon_loss/(num_trajs * numts)
      kld_loss = kld_loss/num_trajs
      spatial_derivative_loss = spatial_derivative_loss/num_trajs
      mean_square_emb = sum_square_emb/num_trajs
      decoder_mean_square_layer_weights = hi.decoder.get_mean_square_layer_weights()
      if scalar_timewarping_lr > 0:
        time_weight_loss = time_weight_loss/(num_trajs * hi.scalar_timewarper.timewarp_parameter_encoder.latent_dim)
      time_endpoint_loss = time_endpoint_loss/num_trajs 
      base_loss = recon_loss.detach().cpu().numpy().copy()
      loss = (recon_loss
              + beta * kld_loss) 

      if curv_loss_penalty_weight > 0:
        loss += curv_loss_penalty_weight * curv_loss
      if decoding_spatial_derivative_regularization > 0:
        loss += decoding_spatial_derivative_regularization * spatial_derivative_loss 

      # only do time_regularization if we've passed a certain amount of training
      if i >= pre_time_learning_epochs and scalar_timewarper_timereg > 0:
        loss += scalar_timewarper_timereg * time_weight_loss
      if i >= pre_time_learning_epochs and scalar_timewarper_endpointreg > 0:
        loss += scalar_timewarper_endpointreg * time_endpoint_loss



      # log to tensorboard
      num_batches_seen += 1
      if num_batches_seen % record_loss_every == 0:
        # 2023-08-29 since we've added dropout, we need to pay attention to whether we're doing train or eval
        hi.eval()

        ####
        ### Huge, basically copypasta, code to check test error:
        ####
        test_raw_recon_loss = torch.tensor(0., dtype=dtype).to(device)
        test_dtw_recon_loss = torch.tensor(0., dtype=dtype).to(device)
        for (test_ts_for_mu, test_ys_for_mu) in test_dataloader:
          estimate, _, _, _= hi.noiseless_forward(test_ys_for_mu, test_ts_for_mu)
          # this truncates timewarped_recon (if needed)
          timewarped_recon = test_time_identity_vector_timewarper.timewarp_first_to_second(estimate, test_ys_for_mu)

          # check error without DTW
          test_raw_recon_loss += nn.functional.mse_loss(timewarped_recon, test_ys_for_mu, reduction="sum")
          
          # Dang, yeah, that was a bug.
          # only truncate (use the output of the IdentityVectorTimewarper) IF we're supposed to.
          # otherwise, you're allowed to use the full vector length in your DTW fit.
          if model_info["vector_timewarper_name"] == "identity_vector_timewarper":
            base_for_warping = timewarped_recon
          else:
            base_for_warping = estimate
          test_dtw_recon, test_dtw_actual = test_time_dtw_vector_timewarper.timewarp_first_and_second(base_for_warping, test_ys_for_mu)
          test_dtw_recon_loss += nn.functional.mse_loss(test_dtw_recon, test_dtw_actual, reduction="sum")
             
        ####
        ### Huge, basically copypasta, code to check noiseless train error with DTW:
        ####
        train_raw_recon_loss = torch.tensor(0., dtype=dtype).to(device)
        train_dtw_recon_loss = torch.tensor(0., dtype=dtype).to(device)
        for (train_ts_for_mu, train_ys_for_mu) in training_dataloader:
          estimate, _, _, _= hi.noiseless_forward(train_ys_for_mu, train_ts_for_mu)
          # this truncates timewarped_recon (if needed)
          timewarped_recon = test_time_identity_vector_timewarper.timewarp_first_to_second(estimate, train_ys_for_mu)
          train_raw_recon_loss += nn.functional.mse_loss(timewarped_recon, train_ys_for_mu, reduction="sum")
          
          #also check error after DTW alignment
          if model_info["vector_timewarper_name"] == "identity_vector_timewarper":
            base_for_warping = timewarped_recon
          else:
            base_for_warping = estimate
          train_dtw_recon, train_dtw_actual = test_time_dtw_vector_timewarper.timewarp_first_and_second(base_for_warping, train_ys_for_mu)
          train_dtw_recon_loss += nn.functional.mse_loss(train_dtw_recon, train_dtw_actual, reduction="sum")

        train_raw_recon_loss /= (num_trajs * numts)
        train_dtw_recon_loss /= (num_trajs * numts)
        test_raw_recon_loss /= (num_test_trajs * numts)
        test_dtw_recon_loss /= (num_test_trajs * numts)


        writer.add_scalar("train/RMSE", np.sqrt(base_loss), num_batches_seen)
        writer.add_scalar("train/learned_dec_log_variance", hi.decoder.log_decoder_variance.detach().cpu().numpy().item(), num_batches_seen)
        writer.add_scalar("train/KLD", kld_loss.detach().cpu().numpy().item(), num_batches_seen)
        writer.add_scalar("train/mean_square_emb", mean_square_emb.detach().cpu().numpy().item(), num_batches_seen)
        writer.add_scalar("train/curv_loss", curv_loss.detach().cpu().numpy().item(), num_batches_seen)
        writer.add_scalar("train/noiselessRMSE", np.sqrt(train_raw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("train/alignedRMSE", np.sqrt(train_dtw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("train/time_weight_loss", time_weight_loss.detach().cpu().numpy(), num_batches_seen)
        writer.add_scalar("train/dec_weight_loss", decoder_mean_square_layer_weights.detach().cpu().numpy(), num_batches_seen)
        writer.add_scalar("train/dec_spatial_derivative_loss", spatial_derivative_loss.detach().cpu().numpy(), num_batches_seen)
        writer.add_scalar("train/time_endpoint_loss", np.sqrt(time_endpoint_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("train/max_decoder_grad_norm", max_decoder_grad_norm.detach().cpu().numpy(), num_batches_seen)
        writer.add_scalar("train/max_encoder_grad_norm", max_encoder_grad_norm.detach().cpu().numpy(), num_batches_seen)
        writer.add_scalar("train/max_timewarp_grad_norm", max_timewarp_grad_norm.detach().cpu().numpy(), num_batches_seen)
        writer.add_scalar("test/noiselessRMSE", np.sqrt(test_raw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("test/alignedRMSE", np.sqrt(test_dtw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        if log_to_wandb_name is not None:
          wandb.log({"num_batches_seen": num_batches_seen,
                     "train_RMSE": np.sqrt(base_loss), 
                     "train/learned_dec_log_variance": hi.decoder.log_decoder_variance.detach().cpu().numpy().item(),
                     "train_KLD": kld_loss.detach().cpu().numpy().item(), 
                     "train/mean_square_emb": mean_square_emb.detach().cpu().numpy().item(),
                     "train_curv_loss": curv_loss.detach().cpu().numpy().item(),
                     "train_noiselessRMSE": np.sqrt(train_raw_recon_loss.detach().cpu().numpy()),
                     "train_alignedRMSE": np.sqrt(train_dtw_recon_loss.detach().cpu().numpy()),
                     "train_time_weight_loss": time_weight_loss.detach().cpu().numpy(),
                     "train/dec_weight_loss": decoder_mean_square_layer_weights.detach().cpu().numpy(),
                     "train/dec_spatial_derivative_loss": spatial_derivative_loss.detach().cpu().numpy(),
                     "train/max_decoder_grad_norm": max_decoder_grad_norm.detach().cpu().numpy(),
                     "train/max_encoder_grad_norm": max_encoder_grad_norm.detach().cpu().numpy(),
                     "train/max_timewarp_grad_norm": max_timewarp_grad_norm.detach().cpu().numpy(),
                     "train_time_endpoint_loss": np.sqrt(time_endpoint_loss.detach().cpu().numpy()),
                     "test_noiselessRMSE": np.sqrt(test_raw_recon_loss.detach().cpu().numpy()),
                     "test_alignedRMSE": np.sqrt(test_dtw_recon_loss.detach().cpu().numpy())})


  np.savez(f"{model_save_dir}/saved_model_info.npz",
      old_saved_model_dir=old_saved_model_dir,
      dtype_string="float" if kwargs["dtype"]==torch.float else "double",
      **kwargs
     )

  encoder_state_dict = hi.encoder.state_dict()
  torch.save(encoder_state_dict, f"{model_save_dir}/encoder_model.pt")
  decoder_state_dict = hi.decoder.state_dict()
  torch.save(decoder_state_dict, f"{model_save_dir}/decoder_model.pt")
  scalar_timewarper_state_dict = hi.scalar_timewarper.state_dict()
  torch.save(scalar_timewarper_state_dict, f"{model_save_dir}/scalar_timewarper_model.pt")
  vector_timewarper_state_dict = vector_timewarper.state_dict()
  torch.save(vector_timewarper_state_dict, f"{model_save_dir}/vector_timewarper_model.pt")
  if log_to_wandb_name is not None:
    run.finish()

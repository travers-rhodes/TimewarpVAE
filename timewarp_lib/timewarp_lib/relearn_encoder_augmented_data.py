#! /usr/env/bin python3
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import numpy as np
import os
import wandb

import timewarp_lib.train_utils as tu
import timewarp_lib.load_model as lm
import timewarp_lib.encoders as et

import timewarp_lib.vector_timewarpers as vtw
import timewarp_lib.parse_model_parameters as pmp

class MockedOptimizer():
  def zero_grad(self):
    pass
  def step(self):
    pass


def initialize_optimizers(hi, emb_lr, emb_eps, useAdam):
  if useAdam:
    emb_optim =    torch.optim.Adam(hi.encoder.parameters(),      lr=emb_lr,        eps=emb_eps)
  else:
    emb_optim =    torch.optim.Adagrad(hi.encoder.parameters(),      lr=emb_lr)
  return emb_optim

def train_model(**kwargs):

  config = dict(**kwargs)
  
  log_to_wandb_name = (kwargs["log_to_wandb_name"] 
                             if "log_to_wandb_name" in kwargs else None)

  device = kwargs["device"]
  random_data_distribution_factor=kwargs["random_data_distribution_factor"]

  saved_model_dir = kwargs["saved_model_dir"]
  modeldatafileobj = np.load(f"{saved_model_dir}/saved_model_info.npz", allow_pickle=True)
  model_info = {key : (modeldatafileobj[key] if key != "initialization_function" else True) for key in modeldatafileobj.keys()}
  loaded_model = lm.LoadedModel(saved_model_dir,device=device)
  hi, vector_timewarper = loaded_model.model, loaded_model.vector_timewarper

  latent_dim = (int)(hi.encoder.latent_dim.item())
  kwargs["latent_dim"] = latent_dim
  config["latent_dim"] = latent_dim

  datafile = kwargs["datafile"]
  dtype = kwargs["dtype"]
  ## Note some assumptions about the model data path also!
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

  datafile = kwargs["datafile"]
  model_save_dir = kwargs["model_save_dir"]
  num_epochs = kwargs["num_epochs"]
  ## Generic
  dtype = kwargs["dtype"]
  encoding_lr = kwargs["encoding_lr"]
  encoding_eps = kwargs["encoding_eps"]
  useAdam = kwargs["useAdam"],
  logname = kwargs["logname"]
  batch_size = (int)(kwargs["batch_size"])
  training_data_timing_noise = (float)(kwargs["training_data_added_timing_noise"])

  ## model_save_dir should NOT already exist
  os.makedirs(model_save_dir)

  hi.encoder = pmp.parse_encoder(**kwargs) 
  hi = hi.to(device)

  if log_to_wandb_name is not None:
    run = wandb.init(project="iclr24-project", entity="XXXXXXXXXX", config=config, reinit=True, group=log_to_wandb_name)

  ### Test Data Section
  ## Load the test data 
  test_data = loaded_data_dict["test"]
  test_ydata = torch.tensor(test_data,dtype=dtype).to(device)
  num_test_trajs, num_test_ts, _ = test_ydata.shape
  test_tdata = torch.tensor(np.linspace(0,1,num_test_ts),dtype=dtype).to(device).expand(num_test_trajs,num_test_ts).unsqueeze(2)
  torch_train_data = torch.utils.data.TensorDataset(test_tdata, test_ydata)
  test_dataloader = torch.utils.data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=False)
  test_time_identity_vector_timewarper = vtw.IdentityVectorTimewarper()
  test_time_dtw_vector_timewarper = vtw.DTWVectorTimewarper()
  ###

  ydata = torch.tensor(training_data,dtype=dtype).to(device)
  num_trajs, numts, traj_channels = ydata.shape

  tdata = torch.tensor(np.linspace(0,1,numts),dtype=dtype).to(device).expand(num_trajs,numts).unsqueeze(2)

  torch_train_data = torch.utils.data.TensorDataset(tdata, ydata)
  training_dataloader = torch.utils.data.DataLoader(torch_train_data, batch_size=batch_size, shuffle=True)

  log_dir = logname
  if log_dir is None:
    raise Exception("Please name your logdir using the logname parameter")
  writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir) #type: ignore (this comment mutes a bug in pyright that thinks we can't access SummaryWriter)
      
  sum_square_noisy_emb = torch.tensor(0.,dtype=dtype).to(device)

  for (training_ts_for_mu, training_ys_for_mu_raw) in training_dataloader:
      hi.eval()
      _, _, _, noisyemb, _, _ = hi.forward_with_noisy_embedding_node(training_ys_for_mu_raw, training_ts_for_mu)
      sum_square_noisy_emb += torch.sum(torch.square(noisyemb))
  mean_square_noisy_emb = sum_square_noisy_emb/num_trajs

 
  embedding_mean = 0. 
  embedding_std = np.sqrt((mean_square_noisy_emb/latent_dim).detach().cpu().numpy()).item()

  emb_optim = initialize_optimizers(hi, encoding_lr, encoding_eps, useAdam)
  num_batches_seen = 0
  record_loss_every = 100

  # for calculation simplicity, only do this once and re-use the retimings
  # (but randomly sample from this pool, of course)
  num_cached_timings = 100000
  cached_timings = []
  # this is a hyperparameter for how many linear knots in our retiming function
  numfirsts = 10
  for _ in range(num_cached_timings):
    warped_xvals = tu.get_warped_xvals(training_data_timing_noise, numfirsts, numts)
    cached_timings.append(warped_xvals)
  cached_timings = np.array(cached_timings)
  print(cached_timings.shape)

  rng = np.random.default_rng()


  for i in range(num_epochs):
      hi.train()
      emb_optim.zero_grad()

      sampled_latents = torch.normal(mean=embedding_mean, std=embedding_std * random_data_distribution_factor, size=(batch_size, latent_dim), dtype=dtype).to(device)
      sampled_ts_indices = rng.choice(num_cached_timings, batch_size)
      sampled_ts = torch.tensor(cached_timings[sampled_ts_indices],dtype=dtype).to(device)
      reconstructed_traj = hi.decoder.decode(sampled_latents, sampled_ts)
      # note: if we want to try this with Transformer Architecture, we shoudl put sampled_ts
      # in here. But for now we don't, so that it's explicit that we (shouldn't) care
      # note that we aren't training logvar, so this new VAE should only be used
      # to generate noiseless embeddings.
      reconstructed_latents, _ = hi.encoder.encode(reconstructed_traj,sampled_ts)

      recon_loss = nn.functional.mse_loss(sampled_latents, reconstructed_latents, reduction="mean")
      recon_loss.backward()
      emb_optim.step()

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


        writer.add_scalar("recon_loss", np.sqrt(recon_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("train/noiselessRMSE", np.sqrt(train_raw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("train/alignedRMSE", np.sqrt(train_dtw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("test/noiselessRMSE", np.sqrt(test_raw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("test/alignedRMSE", np.sqrt(test_dtw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        if log_to_wandb_name is not None:
          wandb.log({"num_batches_seen": num_batches_seen,
                     "recon_loss": np.sqrt(recon_loss.detach().cpu().numpy()),
                     "train_noiselessRMSE": np.sqrt(train_raw_recon_loss.detach().cpu().numpy()),
                     "train_alignedRMSE": np.sqrt(train_dtw_recon_loss.detach().cpu().numpy()),
                     "test_noiselessRMSE": np.sqrt(test_raw_recon_loss.detach().cpu().numpy()),
                     "test_alignedRMSE": np.sqrt(test_dtw_recon_loss.detach().cpu().numpy())})


  np.savez(f"{model_save_dir}/saved_model_info.npz",
      **model_info
     )

  np.savez(f"{model_save_dir}/smoothing_model_info.npz",
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

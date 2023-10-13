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
import timewarp_lib.encoders as et

import timewarp_lib.vector_timewarpers as vtw
import timewarp_lib.parse_model_parameters as pmp

class MockedOptimizer():
  def zero_grad(self):
    pass
  def step(self):
    pass


def initialize_optimizers(hi, time_optim_lr, emb_lr, time_eps, emb_eps, useAdam):
  if useAdam:
    time_optim =  (torch.optim.Adam(hi.scalar_timewarper.parameters(),                lr=time_optim_lr, eps=time_eps)
        if len(list(hi.scalar_timewarper.parameters())) > 0
        else MockedOptimizer())
    emb_optim =    torch.optim.Adam(hi.encoder.parameters(),      lr=emb_lr,        eps=emb_eps)
  else:
    time_optim =   (torch.optim.Adagrad(hi.scalar_timewarper.parameters(), lr=time_optim_lr)
        if len(hi.scalar_timewarper.parameters()) > 0
        else MockedOptimizer())
    emb_optim =    torch.optim.Adagrad(hi.encoder.parameters(),      lr=emb_lr)
  return (time_optim, emb_optim)

def train_model(**kwargs):

  config = dict(**kwargs)
  
  log_to_wandb_name = (kwargs["log_to_wandb_name"] 
                             if "log_to_wandb_name" in kwargs else None)

  device = kwargs["device"]

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


  model_save_dir = kwargs["model_save_dir"]
  num_epochs = kwargs["num_epochs"]
  ## Generic
  scalar_timewarping_lr = kwargs["scalar_timewarping_lr"]
  scalar_timewarping_eps = kwargs["scalar_timewarping_eps"]
  encoding_lr = kwargs["encoding_lr"]
  encoding_eps = kwargs["encoding_eps"]
  useAdam = kwargs["useAdam"],
  logname = kwargs["logname"]
  batch_size = (int)(kwargs["batch_size"])

  ## model_save_dir should NOT already exist
  os.makedirs(model_save_dir)

  hi.encoder = pmp.parse_encoder(**kwargs) 
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
      
  time_optim, emb_optim = initialize_optimizers(hi, scalar_timewarping_lr, encoding_lr, time_eps=scalar_timewarping_eps, emb_eps=encoding_eps, useAdam=useAdam)
  num_batches_seen = 0
  record_loss_every = 100


  for _ in range(num_epochs):
      if scalar_timewarping_lr > 0:
        time_optim.zero_grad()
      emb_optim.zero_grad()
      recon_loss = torch.tensor(0., dtype=dtype).to(device)
      for (test_ts, test_ys) in test_dataloader:
          # 2023-08-29 since we've added dropout, we need to pay attention to whether we're doing train or eval
          hi.train()
          estimate, _,_,_ = hi.noiseless_forward(test_ys, test_ts)
          recon_loss += nn.functional.mse_loss(estimate, test_ys, reduction="sum")

      recon_loss = recon_loss/(num_trajs * numts)
      base_loss = recon_loss.detach().cpu().numpy().copy()
      loss = recon_loss
      loss.backward()

      if scalar_timewarping_lr > 0: 
        time_optim.step()
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


        writer.add_scalar("train/RMSE", np.sqrt(base_loss), num_batches_seen)
        writer.add_scalar("train/noiselessRMSE", np.sqrt(train_raw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("train/alignedRMSE", np.sqrt(train_dtw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("test/noiselessRMSE", np.sqrt(test_raw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        writer.add_scalar("test/alignedRMSE", np.sqrt(test_dtw_recon_loss.detach().cpu().numpy()), num_batches_seen)
        if log_to_wandb_name is not None:
          wandb.log({"num_batches_seen": num_batches_seen,
                     "train_RMSE": np.sqrt(base_loss), 
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

#! /usr/env/bin python3
import numpy as np
import os

#Train a PCA model
# * datafile: 'path_to_data'
# * model_save_dir: 'path_to_full_dir'
# * latent_dim: 3
def train_model(datafile, model_save_dir, latent_dim):
  ## Note some assumptions about the model data path also!
  loaded_data_dict = np.load(datafile)

  ## Load the training data 
  ## assert that the training data is of shape (num_trajs, num_timesteps, num_channels)
  training_data = loaded_data_dict["train"]
  num_trajs, num_timesteps, num_channels = training_data.shape
  print(f"we have num_trajs:{num_trajs}, num_timesteps:{num_timesteps}, num_channels:{num_channels}")

  # for our PCA work, all our work treats trajectories as vectors of length (num_timesteps * num_channels). 
  # So we have
  flattened_trajs = training_data.reshape(num_trajs, num_timesteps * num_channels)

  # compute and remove the mean trajectory (mean trajectory is of shape (num_timesteps * num_channels))
  mean_traj = np.mean(flattened_trajs, axis=0)
  centered_trajs = flattened_trajs - mean_traj[np.newaxis,:]

  # compute the PCA model using SVD
  u, s, vt = np.linalg.svd(centered_trajs)

  # keep only the first latent_dim number of dimensions
  if len(s) < latent_dim:
    raise Exception(f"You can't build a model of dimension {latent_dim} on a dataset with dimensionality {len(s)}")

  # the singular values written in matrix form
  smat = np.diag(s[:latent_dim])
  smatinv = np.diag(1./s[:latent_dim])
  # the first latent_dim directions of variation
  basis_vectors = vt[:latent_dim,:]
  # given a trajectory, use this to find its value in the latent space 
  # note that the dimensions of the embedding matrix are (latent_dim, nt*nc)
  embedding_matrix = smatinv @ basis_vectors 
  # note that we include the s factor here so that our latent space is roughly the unit gaussian ball :brain:
  # note further that the dimensions of the reconstruction matrix are also (latent_dim, nt*nc)
  reconstruction_matrix = smat @ basis_vectors

  # model_save_dir should NOT already exist
  os.makedirs(model_save_dir)
 
  np.savez(f"{model_save_dir}/saved_model_info.npz", 
      modeltype = "pca", 
      modeltypeversion = "0.0", 
      modeltraindatafile= datafile,
      num_timesteps = num_timesteps,
      num_channels = num_channels,
      latent_dim = latent_dim)

  with open(f"{model_save_dir}/saved_model_info.txt", 'w') as f:
    f.write("modeltype: pca; modeltypeverion: 0.0; modeltraindatafile: {datafile}; latent_dim: {latent_dim}\n")
  
  np.savez(f"{model_save_dir}/saved_model.npz", 
      num_trajs = num_trajs, 
      num_timesteps = num_timesteps,
      num_channels = num_channels,
      latent_dim = latent_dim,
      mean_traj = mean_traj,
      embedding_matrix = embedding_matrix,
      reconstruction_matrix = reconstruction_matrix)

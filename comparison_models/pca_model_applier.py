import numpy as np

# Create a ModelApplier object based purely on a
# directory containing model information
class ModelApplier(object):
  def __init__(self, saved_model_dir):
    modelinfo = np.load(f"{saved_model_dir}/saved_model_info.npz")
    modeltype = modelinfo["modeltype"]
    if modeltype != "pca":
      raise Exception(f"This class can only understand pca models, not {modeltype} models")
    modeltypeversion = modelinfo["modeltypeversion"]
    if modeltypeversion != "0.0":
      raise Exception(f"I've versioned my models. This code only understands models of version 0.0. You tried to use a model of version {modeltypeversion}")

    modeldata = np.load(f"{saved_model_dir}/saved_model.npz")
    self.num_trajs = modeldata["num_trajs"]
    self.num_timesteps = modeldata["num_timesteps"]
    self.num_channels = modeldata["num_channels"]
    self.latent_dim = modeldata["latent_dim"]
    self.mean_traj = modeldata["mean_traj"]
    self.embedding_matrix = modeldata["embedding_matrix"]
    self.reconstruction_matrix = modeldata["reconstruction_matrix"]

  # The input data should be of the shape
  # (num_apply_trajs, apply_latent_dim)
  def apply(self, data):
    num_apply_trajs, apply_latent_dim = data.shape
    if apply_latent_dim != self.latent_dim:
      raise Exception(f"The number of latent dim coordinates given: {apply_latent_dim} was not the same as the {self.latent_dim} expected by the model")
  
    # compute the linear combination of basis vectors 
    traj_offset_vectors = data @ self.reconstruction_matrix
    mean_vector = self.mean_traj.reshape(1,self.num_timesteps*self.num_channels)
    # add back the mean vector
    traj_vectors = traj_offset_vectors + mean_vector
    # reshape to the official standard expected shape
    # (num_apply_trajs, num_timesteps, num_channels)
    result_trajs = traj_vectors.reshape(num_apply_trajs, self.num_timesteps, self.num_channels)
    return result_trajs

  # The input data should be of the shape
  # (num_apply_trajs, self.num_timesteps, self.num_channels)
  # The output latent dimensions are of the shape
  # (num_apply_trajs, self.latent_dim)
  def embed(self, data):
    num_apply_trajs, apply_timesteps, apply_num_channels = data.shape
    assert apply_timesteps == self.num_timesteps, "timesteps must match for PCA"
    assert apply_num_channels == self.num_channels, "channels must match for PCA"
     
    # for our PCA work, all our work treats trajectories as vectors of length (num_timesteps * num_channels). 
    # So we have
    flattened_trajs = data.reshape(num_apply_trajs, self.num_timesteps * self.num_channels)
    mean_vector = self.mean_traj.reshape(1,self.num_timesteps*self.num_channels)
    centered_trajs = flattened_trajs - mean_vector
    # the dimensions of the embedding matrix are (latent_dim, nt*nc)
    latent_vals_mat = centered_trajs @ self.embedding_matrix.T
    return(latent_vals_mat)

import numpy as np
import comparison_models.pca_model_applier

# Create a ModelApplier object based purely on a
# directory containing model information
class ModelApplier(object):
  # ok this is maybe too magical...
  def __new__(cls, saved_model_dir):
    modelinfo = np.load(f"{saved_model_dir}/saved_model_info.npz")
    modeltype = modelinfo["modeltype"]
    if modeltype == "pca":
      return comparison_models.pca_model_applier.ModelApplier(saved_model_dir)
    else:
      raise Exception("I don't know how to apply model type {modeltype}")
    

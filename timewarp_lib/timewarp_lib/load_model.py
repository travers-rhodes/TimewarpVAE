import numpy as np
import torch

import timewarp_lib.parse_model_parameters as pmp

#Load a trained model into
# self.encoder, self.decoder, self.scalar_timewarper, self.vector_timewarper
class LoadedModel(object):
  def __init__(self, saved_model_dir, device="cpu"):
    modeldatafileobj = np.load(f"{saved_model_dir}/saved_model_info.npz",allow_pickle=True)
    modeldata = {key : (modeldatafileobj[key] if key != "initialization_function" else True) for key in modeldatafileobj.keys()}
    modeldata["dtype"] = torch.float if modeldata["dtype_string"]=="float" else torch.double
    self.modeldata = modeldata

    self.model, self.vector_timewarper = pmp.parse_arguments(**modeldata)

    encoder_state_dict = torch.load(f"{saved_model_dir}/encoder_model.pt", map_location=torch.device(device))
    decoder_state_dict = torch.load(f"{saved_model_dir}/decoder_model.pt", map_location=torch.device(device))
    scalar_timewarper_state_dict = torch.load(f"{saved_model_dir}/scalar_timewarper_model.pt", map_location=torch.device(device))
    vector_timewarper_state_dict = torch.load(f"{saved_model_dir}/vector_timewarper_model.pt", map_location=torch.device(device))

    self.model.encoder.load_state_dict(encoder_state_dict)
    self.model.decoder.load_state_dict(decoder_state_dict)
    self.model.scalar_timewarper.load_state_dict(scalar_timewarper_state_dict)
    self.vector_timewarper.load_state_dict(vector_timewarper_state_dict)

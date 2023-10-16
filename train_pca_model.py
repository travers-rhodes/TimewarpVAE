#! /usr/bin/env python3

import comparison_models.pca_model_trainer as pmt
from datetime import datetime

SCRATCHFOLDER="results/augrespca"
DATAFILE=f"data/trainTest2DLetterACache.npz"

for latent_dim in range(30): 
  timestr = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
  modelsavedir=f"{SCRATCHFOLDER}/{timestr}/savedmodel"
  pmt.train_model(DATAFILE, modelsavedir, latent_dim)

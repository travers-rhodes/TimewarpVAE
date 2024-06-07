#! /usr/bin/env python3

import timewarp_lib.load_model as lm
import numpy as np
import torch
import timewarp_lib.vector_timewarpers as vtw
import torch.nn as nn
import pandas as pd
import json
import timewarp_lib.decoders as d

def clean(rawdata):
    return rawdata.replace("'","\"").replace("False","0").replace("True","1").replace("None","\"None\"")

def get_all_data():
    rawdata1 = pd.read_csv("../project.csv")
    rawdata1 = rawdata1.reset_index()  # make sure indexes pair with number of rows

    rawdata2 = pd.read_csv("../projectrateinvariantvae.csv")
    rawdata2 = rawdata2.reset_index()  # make sure indexes pair with number of rows


    rawdata = pd.concat([rawdata1,rawdata2])
    rawdata.reset_index()
    return rawdata

rawdata = get_all_data()

# we trained on augmented data, but we should just apply to regular data

DATAFILE=f"../data/trainTest2DLetterARescaled.npz"
data = np.load(DATAFILE)
test = data["test"]
train = data["train"]


num_trains, num_ts, channels = train.shape
num_tests, num_ts, channels = test.shape
train_ts = torch.tensor(np.linspace(0,1,num_ts),dtype=torch.float).expand(num_trains,num_ts).unsqueeze(2)
test_ts = torch.tensor(np.linspace(0,1,num_ts),dtype=torch.float).expand(num_tests,num_ts).unsqueeze(2)

train_torch = torch.tensor(train,dtype=torch.float)
test_torch = torch.tensor(test,dtype=torch.float)

def no_timewarping_recon(hi, traj, ts):
    mu, _ = hi.model.encoder.encode(traj)
    if type(hi.model.decoder) is d.RateInvariantDecoder:
        T = hi.model.decoder.T
        fillval = 1/np.sqrt(T)
        mu[:,:T] = fillval
    
    recontraj = hi.model.decoder.decode(mu,ts)
    return recontraj

test_time_dtw_vector_timewarper = vtw.DTWVectorTimewarper()

results = []
for index, row in rawdata.iterrows():
    configdict = json.loads(clean(row.config))
    prefix = ""
    if configdict["model_save_dir"][:3] == "../":
        load_model_dir = "../copiedResults/" + configdict["model_save_dir"][3:]
    else:
        load_model_dir = "../copiedResults/" + configdict["model_save_dir"]
    try:
        hi = lm.LoadedModel(load_model_dir)
    except:
        print(f"Couldn't load model {load_model_dir}")
        continue
    recon_train = no_timewarping_recon(hi, train_torch, train_ts).detach().numpy()
    recon_test = no_timewarping_recon(hi, test_torch, test_ts).detach().numpy()
    train_dtw_recon, train_dtw_actual = test_time_dtw_vector_timewarper.timewarp_first_and_second(
        torch.tensor(recon_train,dtype=torch.float), 
        torch.tensor(train,dtype=torch.float))
    train_aligned_loss = np.sqrt(
        nn.functional.mse_loss(train_dtw_recon, train_dtw_actual, reduction="sum").detach().numpy()
        / (num_ts * num_trains))
    test_dtw_recon, test_dtw_actual = test_time_dtw_vector_timewarper.timewarp_first_and_second(
        torch.tensor(recon_test,dtype=torch.float), 
        torch.tensor(test,dtype=torch.float))
    test_aligned_loss = np.sqrt(
        nn.functional.mse_loss(test_dtw_recon, test_dtw_actual, reduction="sum").detach().numpy()
        / (num_ts * num_tests))
    print(f"Ran model {load_model_dir}")
    results.append((load_model_dir,configdict["model_save_dir"],train_aligned_loss,test_aligned_loss))
    
import pickle
with open('aligned_reconstruction_wo_tw_errors.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

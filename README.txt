##
Install needed packages using something like
```
conda create --name timewarpvae --file conda_requirements.txt
conda activate timewarpvae
pip install -r pip_requirements.txt
```

In `timewarp_lib` directory, run the `./make.sh` command, or the `./make_nocuda.sh` if you do not have CUDA installed.

In this directory, run `make` to download and process data

For the experiments in our paper, we ran
train_model.py
train_model_no_tw.py
train_fork_conv.py
train_fork_model.py
train_fork_notw_model.py
train_model_notw_DTW.py

created pca model on timing-augmented data using
train_pca_model.py
applied it using

Also ran a single model from
./train_model_notw_trans.py
before cancelling because it took 4.6 hours and was slowing down other runs.

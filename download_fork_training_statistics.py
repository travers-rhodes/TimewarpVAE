#!/usr/bin/env python3
import pandas as pd
import wandb

for groupname, savename in (("retrainedforkdata", "fork_project.csv"),("forkrateinvariantvae","forkprojectrateinvariantvae.csv")):
  api = wandb.Api()
  entity, project = "teamANONYMOUS", "anonymous-project"
  runs = api.runs(entity + "/" + project, {"group": groupname})
  
  summary_list, config_list, name_list = [], [], []
  for run in runs:
      # .summary contains output keys/values for
      # metrics such as accuracy.
      #  We call ._json_dict to omit large files
      summary_list.append(run.summary._json_dict)
  
      # .config contains the hyperparameters.
      #  We remove special values that start with _.
      config_list.append(
          {k: v for k,v in run.config.items()
           if not k.startswith('_')})
  
      # .name is the human-readable name of the run.
      name_list.append(run.name)
  
  runs_df = pd.DataFrame({
      "summary": summary_list,
      "config": config_list,
      "name": name_list
      })
  
  runs_df.to_csv(savename)

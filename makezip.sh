#!/bin/bash
zip timewarpVAECode.zip -r * -x ".git/*" "makezip.sh" "timewarpVAECode.zip" "paper_calculations/dmpmodels/*" "paper_calculations/aligned_reconstruction_wo_tw_errors.pickle" "paper_images/forkTipFromTurboSquid.stl" "*/.ipynb_checkpoints" "*.csv" "paper_images/*.png" "paper_images/*.pdf" "raw_fork_trajectory_data/fork_trajectory_recordings.zip"

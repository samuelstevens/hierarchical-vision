#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=PAS2136
#SBATCH --gpus-per-node=4
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=32


# Activate virtual environment
source $HOME/projects/hierarchical-vision/venv/bin/activate

# Run job
composer main.py \
  --base $BASE_CONFIG_FILE \
  --machine $MACHINE_CONFIG_FILE \
  --extra $EXTRA_CONFIG_FILES \
  --exp $EXP_CONFIG_FILE

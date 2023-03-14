#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=PAS2136
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=32


# Activate virtual environment
source $HOME/projects/hierarchical-vision/venv/bin/activate

python -m tools.rand_split --kind species --frac 0.2 \
  --input /fs/scratch/PAS2136/hierarchical-vision/datasets/inat21/raw \
  --output /fs/scratch/PAS2136/hierarchical-vision/datasets/inat21/rand-species-split

python -m tools.rand_split --kind genus --frac 0.2 \
  --input /fs/scratch/PAS2136/hierarchical-vision/datasets/inat21/raw \
  --output /fs/scratch/PAS2136/hierarchical-vision/datasets/inat21/rand-genus-split

python -m tools.rand_split --kind subtree --frac 0.2 \
  --input /fs/scratch/PAS2136/hierarchical-vision/datasets/inat21/raw \
  --output /fs/scratch/PAS2136/hierarchical-vision/datasets/inat21/rand-subtree-split

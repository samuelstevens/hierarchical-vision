## Getting Started

1. Use Python 3.10.9: `pyenv loacl 3.10.9`
2. Use a virtual environment: `python -m venv venv && source venv/bin/activate.fish`
3. Install requirements: `pip install -r requirements.txt`

## Make Pretraining Splits

```py
python -m tools.rand_split --kind species --input /local/scratch/cv_datasets/inat21/raw --output /local/scratch/cv_datasets/inat21/rand-species-split --frac 0.2
python -m tools.rand_split --kind genus --input /local/scratch/cv_datasets/inat21/raw --output /local/scratch/cv_datasets/inat21/rand-genus-split --frac 0.2
python -m tools.rand_split --kind subtree --input /local/scratch/cv_datasets/inat21/raw --output /local/scratch/cv_datasets/inat21/rand-subtree-split --frac 0.2
```

## Do Pretraining

We only did pretraining hyperparameter sweeps on the rand-species task.

Once we have decided on hyperparameters and written them to configs/pretrain/fixed, we have to vary the datasplit and the pretraining objective (cross entropy, multitask, hierarchical cross entropy).
In fish:

```fish
for split in species genus subtree
    for objective in cross_entropy multitask hxe
        composer main.py \
          --machine configs/machine/strawberry0.yaml \
          --exp \
            configs/pretrain/inat21.yaml \
            configs/pretrain/inat21_rand_$split.yaml \
            configs/pretrain/r50_fast.yaml \
            configs/pretrain/fixed/r50_pretrain_$objective.yaml
    done
done
```

With slurm:
```sh
python submit_slurm.py \
  --machine configs/machines/ascend.yaml \
  --exp \
    configs/pretrain/inat21.yaml \
    configs/pretrain/inat21_$split.yaml \
    configs/pretrain/r50_fast.yaml \
    configs/pretrain/fixed/r50_pretrain_$objective.yaml \
  --submit
```

Then we can do the downstream tasks.


## Make Low Data Splits

```py
python -m tools.low_data_split --input /local/scratch/cv_datasets/inat21/rand-species-split/downstream/ --output /local/scratch/cv_datasets/inat21/rand-species-split/downstream-1shot --shots 1
python -m tools.low_data_split --input /local/scratch/cv_datasets/inat21/rand-species-split/downstream/ --output /local/scratch/cv_datasets/inat21/rand-species-split/downstream-5shot --shots 5
python -m tools.low_data_split --input /local/scratch/cv_datasets/inat21/rand-species-split/downstream/ --output /local/scratch/cv_datasets/inat21/rand-species-split/downstream-10shot --shots 10

python -m tools.low_data_split --input /local/scratch/cv_datasets/inat21/rand-genus-split/downstream/ --output /local/scratch/cv_datasets/inat21/rand-genus-split/downstream-1shot --shots 1
python -m tools.low_data_split --input /local/scratch/cv_datasets/inat21/rand-genus-split/downstream/ --output /local/scratch/cv_datasets/inat21/rand-genus-split/downstream-5shot --shots 5
python -m tools.low_data_split --input /local/scratch/cv_datasets/inat21/rand-genus-split/downstream/ --output /local/scratch/cv_datasets/inat21/rand-genus-split/downstream-10shot --shots 10

python -m tools.low_data_split --input /local/scratch/cv_datasets/inat21/rand-subtree-split/downstream/ --output /local/scratch/cv_datasets/inat21/rand-subtree-split/downstream-1shot --shots 1
python -m tools.low_data_split --input /local/scratch/cv_datasets/inat21/rand-subtree-split/downstream/ --output /local/scratch/cv_datasets/inat21/rand-subtree-split/downstream-5shot --shots 5
python -m tools.low_data_split --input /local/scratch/cv_datasets/inat21/rand-subtree-split/downstream/ --output /local/scratch/cv_datasets/inat21/rand-subtree-split/downstream-10shot --shots 10
```

## Run Low Data Experiments

```py
composer main.py \
  --machine configs/machine/strawberry0.yaml \
  --exp \
    configs/downstream/rand_species_10shot.yaml \
    configs/linear_probe/r50_base.yaml \
    configs/linear_probe/r50_rand_species_cross_entropy.yaml

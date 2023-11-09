#!/bin/bash
#SBATCH --job-name=pytest_dist_slurm
#SBATCH --ntasks=3
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_tests/outputs/test_dist.out

export TORCHRUNX_HOME=$HOME/torchrunx
srun --mpi=pmi2 python3 -m pytest -v $TORCHRUNX_HOME/tests/test_dist.py

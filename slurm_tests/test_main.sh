#!/bin/bash
#SBATCH --job-name=pytest_main_slurm
#SBATCH --ntasks=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_tests/outputs/test_main.out

export TORCHRUNX_HOME=$HOME/torchrunx
srun --mpi=pmi2 python3 -m pytest -v $TORCHRUNX_HOME/tests/test_main.py

#!/bin/bash
#SBATCH --job-name=pytest_multinode_spawn_slurm
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_tests/outputs/test_multinode_spawn.out

export TORCHRUNX_HOME=$HOME/torchrunx
srun --mpi=pmi2 python3 -m pytest -v $TORCHRUNX_HOME/tests/test_multinode_spawn.py

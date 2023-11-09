import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

@pytest.fixture(scope="module")
def setup_slurm_env():
    # Assuming SLURM environment variables are set, for example:
    # SLURM_PROCID: The rank of the process
    # SLURM_NTASKS: The total number of tasks
    os.environ['MASTER_ADDR'] = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = '29500'  # Use a fixed port or generate dynamically
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    backend = 'gloo'  # or 'nccl' if using GPUs
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    yield
    dist.destroy_process_group()

def run_all_reduce(rank, world_size):
    """ Run collective communication. """
    group = dist.new_group([0, 1, 2])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    assert tensor[0] == world_size, f"Rank {rank} failed the all_reduce operation."

@pytest.mark.parametrize("rank", [0, 1, 2])
def test_all_reduce(setup_slurm_env, rank):
    world_size = int(os.environ['SLURM_NTASKS'])
    run_all_reduce(rank, world_size)
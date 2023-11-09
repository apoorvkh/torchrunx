import os
import tempfile
from typing import Callable

import dill
import torch.distributed as dist
# import tyro # what does this do
from torch.distributed.elastic.multiprocessing import start_processes
from torch.distributed.elastic.multiprocessing.api import MultiprocessContext, RunProcsResult


def entrypoint(index: int, fn: Callable, *args):
    os.environ["LOCAL_RANK"] = str(index)
    os.environ["RANK"] = str(index)
    dist.init_process_group(backend="nccl", world_size=int(os.environ["WORLD_SIZE"]), rank=index)
    return fn(*args)

def main(controller_ip: str, controller_port: int):

    # TODO initialize communication with controller
    # torch.distributed.init_process_group(gloo), ranks >= 1

    # TODO: get serialized function and arguments
    serialized_function: str = ""
    fn = dill.loads(serialized_function)

    # TODO: get node, process, rank information
    num_nodes: int = 1
    num_processes: int = 1
    master_ip: str = ""
    master_port: int = 0
    rank: int = 0

    os.environ["WORLD_SIZE"] = str(num_nodes * num_processes)
    os.environ["NODE_RANK"] = str(rank)
    os.environ["NPROC"] = str(num_processes)
    os.environ["MASTER_ADDR"] = master_ip
    os.environ["MASTER_PORT"] = str(master_port)

    args = {i: tuple() for i in range(num_processes)}
    envs = {i: {} for i in range(num_processes)}

    log_dir = None
    if log_dir is None:
        log_dir = tempfile.mkdtemp()
        
    ctx: MultiprocessContext = start_processes(
        name="distributed_function",
        entrypoint=entrypoint,
        args={i: (i, fn, *args[i]) for i in args},
        envs=envs,
        log_dir=log_dir,
        start_method="spawn",
    )
    result: RunProcsResult = ctx.wait()
    return result.return_values


if __name__ == "__main__":
    tyro.cli(main)

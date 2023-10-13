import os
import socket
import tempfile
from contextlib import closing
from typing import Callable, Dict, Optional, Tuple, Union

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import start_processes
from torch.distributed.elastic.multiprocessing.api import MultiprocessContext, RunProcsResult


def entrypoint(index: int, fn: Callable, *args):
    os.environ["LOCAL_RANK"] = str(index)
    os.environ["RANK"] = str(index)
    dist.init_process_group(backend="nccl", world_size=int(os.environ["WORLD_SIZE"]), rank=index)
    return fn(*args)


def run_distributed(
    fn: Callable,
    args: Union[None, Tuple, Dict[int, Tuple]] = None,
    envs: Union[None, Dict[str, str], Dict[int, Dict[str, str]]] = None,
    log_dir: Optional[str] = None,
    num_nodes: int = 1,
    num_procs: int = 1,
):
    assert num_nodes == 1, "only implemented for localhost"

    os.environ["WORLD_SIZE"] = str(num_nodes * num_procs)
    os.environ["NODE_RANK"] = "0"
    os.environ["NPROC"] = str(num_procs)

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        random_port = s.getsockname()[1]

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(random_port)

    if args is None:
        args = {i: tuple() for i in range(num_procs)}
    elif isinstance(args, Tuple):
        args = {i: args for i in range(num_procs)}

    if envs is None:
        envs = {i: {} for i in range(num_procs)}
    elif isinstance(next(iter(envs)), int) is False:
        envs = {i: envs for i in range(num_procs)}

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

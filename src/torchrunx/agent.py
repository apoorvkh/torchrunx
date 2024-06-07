import socket
import os
import tempfile
import torch.distributed as dist

# import tyro # what does this do
from torch.distributed.elastic.multiprocessing import start_processes, DefaultLogsSpecs
from torch.distributed.elastic.multiprocessing.api import MultiprocessContext, Std
from datetime import timedelta

from torchrunx.utils import Serializable, get_open_port, broadcast, gather, all_gather
from torchrunx.launcher import LaunchConfig, AgentStatus

from dataclasses import dataclass
from typing import Callable
import torch


@dataclass
class WorkerArgs(Serializable):
    function: Callable
    master_ip: str
    master_port: int
    backend: str


def entrypoint(serialized_worker_args: bytes, *args):
    worker_args = WorkerArgs.from_serialized(serialized_worker_args)

    fn = worker_args.function
    master_ip = worker_args.master_ip
    master_port = worker_args.master_port
    backend = worker_args.backend

    # Initialize TCPStore for group
    is_master = os.environ["RANK"] == "0"
    world_size = int(os.environ["WORLD_SIZE"])
    store = dist.TCPStore(
        master_ip, master_port, world_size=world_size, is_master=is_master
    )

    if backend is None:
        backend = "gloo|nccl" if torch.cuda.is_available() else "gloo"
    rank = int(os.environ["RANK"])
    dist.init_process_group(
        backend=backend, world_size=world_size, rank=rank, store=store
    )
    return fn(*args)


def main(world_size: int, rank: int, launcher_ip: str, launcher_port: int):
    # create client TCPStore for initializing launcher-agent process group
    store = dist.TCPStore(launcher_ip, launcher_port)
    # print("got store, trying setup")
    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        store=store,
        timeout=timedelta(seconds=30),
    )

    # receieve parameters from launcher
    config: LaunchConfig = broadcast(object=None, src=0)

    worker_world_size = config.world_size
    worker_ranks = config.node_worker_ranks[rank - 1]
    num_workers = len(worker_ranks)

    # broadcast/receive launcher worker's IP and port
    if rank == 1:
        # rank 1 agent is responsible for rank 0 worker, aka the "master worker"
        # thus grab a port on this agent's node
        master_hostname = socket.gethostname()

        master_ip = socket.gethostbyname(master_hostname)
        master_port = get_open_port()
        master = (master_ip, master_port)
    else:
        # else, we listen for the broadcast
        master = (None, None)

    master_ip, master_port = broadcast(object=master, src=1)

    # send process pid to launcher
    gather(object=os.getpid(), dst=0)

    # set arguments and environmental variables for each worker
    # args = {i: arguments for i in range(num_processes)}
    envs = {
        i: {
            "RANK": str(worker_ranks[i]),
            "LOCAL_RANK": str(i),
            "WORLD_SIZE": str(worker_world_size),
        }
        for i in range(num_workers)
    }

    # logging directory
    log_dir = None
    if log_dir is None:
        log_dir = tempfile.mkdtemp()  #  f"/users/pcurtin1/torchrunx/log/{rank}/" #

    worker_args = WorkerArgs(function=config.fn, master_ip=master_ip, master_port=master_port, backend=config.backend)
    serialized_worker_args = worker_args.serialized

    # spawn workers
    ctx: MultiprocessContext = start_processes(
        name="distributed_function",
        entrypoint=entrypoint,
        args={
            i: (serialized_worker_args,) for i in range(num_workers)
        },
        envs=envs,
        logs_specs=DefaultLogsSpecs(log_dir=log_dir, redirects=Std.ALL),
        start_method="spawn",
    )
    done = False
    while True:
        # determine status of this agent, five-second timeout
        if not done:
            result = ctx.wait(5)
        status = AgentStatus(result)
        done = status.is_done()
        # grab statuses of other agents
        statuses: list[AgentStatus]
        try:
            statuses = all_gather(object=status)
        except:
            ctx.close()
            return
        # if any workers on any agent have failed
        if any(map(lambda s: s.is_failed(), statuses)):
            # terminate local workers and exit
            ctx.close()
            return

        # else, check if everything's done
        if all(map(lambda s: s.is_done(), statuses)):
            # we can exit loop and gather return values
            break

        # otherwise, continue...

    return_values = {worker_ranks[k]: v for k, v in result.return_values.items()}
    gather(object=return_values, dst=0)

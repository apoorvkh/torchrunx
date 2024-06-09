from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Literal

import cloudpickle
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from torch.distributed.elastic.multiprocessing.api import MultiprocessContext, RunProcsResult, Std
from typing_extensions import Self

from .utils import AgentStatus, LauncherAgentGroup


@dataclass
class WorkerArgs:
    function: Callable
    master_ip: str
    master_port: int
    backend: Literal["mpi", "gloo", "nccl", "ucc", None]

    def to_bytes(self) -> bytes:
        return cloudpickle.dumps(self)

    @classmethod
    def from_bytes(cls, serialized: bytes) -> Self:
        return cloudpickle.loads(serialized)


def entrypoint(serialized_worker_args: bytes, *args):
    worker_args = WorkerArgs.from_bytes(serialized_worker_args)

    fn = worker_args.function
    master_ip = worker_args.master_ip
    master_port = worker_args.master_port
    backend = worker_args.backend

    # Initialize TCPStore for group
    is_master = os.environ["RANK"] == "0"
    world_size = int(os.environ["WORLD_SIZE"])
    store = dist.TCPStore(master_ip, master_port, world_size=world_size, is_master=is_master)

    if backend is None:
        backend = "gloo|nccl" if torch.cuda.is_available() else "gloo"
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank, store=store)
    return fn(*args)


def main(world_size: int, rank: int, launcher_ip: str, launcher_port: int):
    launcher_group = LauncherAgentGroup(
        world_size=world_size,
        rank=rank,
        launcher_hostname=launcher_ip,
        launcher_port=launcher_port,
    )

    # receieve parameters from launcher
    config = launcher_group.recv_launch_config()
    worker_world_size = config.world_size
    worker_ranks = config.node_worker_ranks[rank - 1]
    num_workers = len(worker_ranks)

    main_agent_ip, main_agent_port = launcher_group.sync_main_agent_ip_port()
    launcher_group.send_process_id()

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

    worker_args = WorkerArgs(
        function=config.fn,
        master_ip=main_agent_ip,
        master_port=main_agent_port,
        backend=config.backend,
    )
    serialized_worker_args = worker_args.to_bytes()

    # spawn workers
    ctx: MultiprocessContext = start_processes(  # pyright: ignore[reportAssignmentType]
        name="distributed_function",
        entrypoint=entrypoint,
        args={i: (serialized_worker_args,) for i in range(num_workers)},
        envs=envs,
        logs_specs=DefaultLogsSpecs(log_dir=log_dir, redirects=Std.ALL),
        start_method="spawn",
    )

    status = AgentStatus()
    result: RunProcsResult | None = None
    while True:
        if result is None:
            result = ctx.wait(5)
            status = AgentStatus.from_result(result=result, worker_ranks=worker_ranks)

        try:
            agent_statuses = launcher_group.sync_agent_statuses(status=status)
            if any([s.is_failed() for s in agent_statuses]):
                raise RuntimeError()
        except:
            ctx.close()
            return

        if all([s.is_done() for s in agent_statuses]):
            break

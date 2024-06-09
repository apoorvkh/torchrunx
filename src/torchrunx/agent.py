from __future__ import annotations

import os
import socket
import tempfile
from dataclasses import dataclass
from typing import Callable, Literal

import cloudpickle
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs, start_processes
from torch.distributed.elastic.multiprocessing.api import MultiprocessContext, RunProcsResult, Std
from typing_extensions import Self

from .utils import AgentPayload, AgentStatus, LauncherAgentGroup, LauncherPayload, get_open_port


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

    payload = AgentPayload(
        ip=socket.gethostbyname(socket.gethostname()),
        port=get_open_port(),
        process_id=os.getpid(),
    )

    all_payloads = launcher_group.sync_payloads(payload=payload)
    launcher_payload: LauncherPayload = all_payloads[0]  # pyright: ignore[reportAssignmentType]
    main_agent_payload: AgentPayload = all_payloads[1]  # pyright: ignore[reportAssignmentType]

    worker_world_size = launcher_payload.worker_world_size
    worker_global_ranks = launcher_payload.worker_global_ranks[rank - 1]
    num_workers = len(worker_global_ranks)

    # logging directory
    log_dir = None
    if log_dir is None:
        log_dir = tempfile.mkdtemp()  #  f"/users/pcurtin1/torchrunx/log/{rank}/" #

    serialized_worker_args = WorkerArgs(
        function=launcher_payload.fn,
        master_ip=main_agent_payload.ip,
        master_port=main_agent_payload.port,
        backend=launcher_payload.backend,
    ).to_bytes()

    args = {i: (serialized_worker_args,) for i in range(num_workers)}

    envs = {
        i: {
            "RANK": str(worker_global_ranks[i]),
            "LOCAL_RANK": str(i),
            "WORLD_SIZE": str(worker_world_size),
        }
        for i in range(num_workers)
    }

    # spawn workers
    ctx: MultiprocessContext = start_processes(  # pyright: ignore[reportAssignmentType]
        name="distributed_function",
        entrypoint=entrypoint,
        args=args,
        envs=envs,
        logs_specs=DefaultLogsSpecs(log_dir=log_dir, redirects=Std.ALL),
        start_method="spawn",
    )

    status = AgentStatus()
    result: RunProcsResult | None = None
    while True:
        if result is None:
            result = ctx.wait(5)
            status = AgentStatus.from_result(result=result, worker_global_ranks=worker_global_ranks)

        try:
            agent_statuses = launcher_group.sync_agent_statuses(status=status)
            if any([s.is_failed() for s in agent_statuses]):
                raise RuntimeError()
        except:
            ctx.close()
            return

        if all([s.is_done() for s in agent_statuses]):
            break

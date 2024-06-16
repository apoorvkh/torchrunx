from __future__ import annotations

import os
import socket
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cloudpickle
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
from torch.distributed.elastic.multiprocessing.api import MultiprocessContext, Std
from typing_extensions import Self

from .utils import (
    AgentPayload,
    AgentStatus,
    LauncherAgentGroup,
    LauncherPayload,
    get_open_port,
)


@dataclass
class WorkerArgs:
    function: Callable
    master_ip: str
    master_port: int
    backend: Literal["mpi", "gloo", "nccl", "ucc", None]
    log_dir: str

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
    log_dir = worker_args.log_dir

    rank = int(os.environ["RANK"])

    log_file = open(Path(log_dir) / f"worker_{rank}.log", "w")
    with redirect_stderr(log_file), redirect_stdout(log_file):
        is_master = os.environ["RANK"] == "0"
        world_size = int(os.environ["WORLD_SIZE"])
        store = dist.TCPStore(master_ip, master_port, world_size=world_size, is_master=is_master)  # pyright: ignore[reportPrivateImportUsage]

        if backend is None:
            backend = "gloo|nccl" if torch.cuda.is_available() else "gloo"

        dist.init_process_group(backend=backend, world_size=world_size, rank=rank, store=store)
        return fn(*args)


def main(world_size: int, rank: int, launcher_ip: str, launcher_port: int, log_dir: str):
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

    serialized_worker_args = WorkerArgs(
        function=launcher_payload.fn,
        master_ip=main_agent_payload.ip,
        master_port=main_agent_payload.port,
        backend=launcher_payload.backend,
        log_dir=log_dir,
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

    tee = {i: (Std.ALL if i == 0 else Std.NONE) for i in range(num_workers)}

    ctx = MultiprocessContext(
        name="distributed_function",
        entrypoint=entrypoint,
        args=args,
        envs=envs,
        logs_specs=DefaultLogsSpecs(log_dir=None, tee=tee),
        start_method="spawn",
    )

    try:
        ctx.start()

        status = AgentStatus()
        while True:
            if status.is_running():
                status = AgentStatus.from_result(
                    result=ctx.wait(5), worker_global_ranks=worker_global_ranks
                )

            agent_statuses = launcher_group.sync_agent_statuses(status=status)

            if any(s.is_failed() for s in agent_statuses):
                raise RuntimeError()
            elif all(s.is_done() for s in agent_statuses):
                break
    except Exception:
        ctx.close()
        raise

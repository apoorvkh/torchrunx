from __future__ import annotations

import datetime
import logging
import logging.handlers
import os
import socket
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Literal

import cloudpickle
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
from torch.distributed.elastic.multiprocessing.api import MultiprocessContext
from typing_extensions import Self

from .log_utils import RenamingSocketHandler, StreamLogger
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
    master_hostname: str
    master_port: int
    backend: Literal["mpi", "gloo", "nccl", "ucc", None]
    rank: int
    local_rank: int
    local_world_size: int
    world_size: int
    hostname: str
    log_host: str
    log_port: int
    timeout: int

    def to_bytes(self) -> bytes:
        return cloudpickle.dumps(self)

    @classmethod
    def from_bytes(cls, serialized: bytes) -> Self:
        return cloudpickle.loads(serialized)


class WorkerTee(object):
    def __init__(self, name: os.PathLike | str, mode: str):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.__del__()

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


def entrypoint(serialized_worker_args: bytes):
    worker_args = WorkerArgs.from_bytes(serialized_worker_args)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.name = (
        f"torchrunx.{worker_args.hostname}[{worker_args.local_rank}]"  # overwrite root logger name
    )
    socketHandler = RenamingSocketHandler(worker_args.log_host, worker_args.log_port, logger.name)
    logger.addHandler(socketHandler)

    sys.stdout = StreamLogger(logger, sys.__stdout__)
    sys.stderr = StreamLogger(logger, sys.__stderr__)

    store = dist.TCPStore(  # pyright: ignore[reportPrivateImportUsage]
        host_name=worker_args.master_hostname,
        port=worker_args.master_port,
        world_size=worker_args.world_size,
        is_master=(worker_args.rank == 0),
    )

    backend = worker_args.backend
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    logging.debug(f"using backend: {backend}")

    dist.init_process_group(
        backend=backend,
        world_size=worker_args.world_size,
        rank=worker_args.rank,
        store=store,
        timeout=datetime.timedelta(seconds=worker_args.timeout),
    )

    os.environ["RANK"] = str(worker_args.rank)
    os.environ["LOCAL_RANK"] = str(worker_args.local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(worker_args.local_world_size)
    os.environ["WORLD_SIZE"] = str(worker_args.world_size)
    os.environ["MASTER_ADDR"] = worker_args.master_hostname
    os.environ["MASTER_PORT"] = str(worker_args.master_port)

    logging.debug(f"executing function: {worker_args.function}")
    return worker_args.function()


def main(launcher_agent_group: LauncherAgentGroup):
    agent_rank = launcher_agent_group.rank - 1

    payload = AgentPayload(
        hostname=socket.getfqdn(),
        port=get_open_port(),
        process_id=os.getpid(),
    )

    all_payloads = launcher_agent_group.sync_payloads(payload=payload)
    launcher_payload: LauncherPayload = all_payloads[0]  # pyright: ignore[reportAssignmentType]
    main_agent_payload: AgentPayload = all_payloads[1]  # pyright: ignore[reportAssignmentType]

    hostname = launcher_payload.hostnames[agent_rank]
    worker_world_size = launcher_payload.worker_world_size
    worker_global_ranks = launcher_payload.worker_global_ranks[agent_rank]
    num_workers = len(worker_global_ranks)

    logger = logging.getLogger(f"torchrunx.{launcher_payload.hostnames[agent_rank]}")
    logger.setLevel(logging.DEBUG)
    socketHandler = logging.handlers.SocketHandler(
        launcher_payload.log_host,
        launcher_payload.log_port,
    )
    logger.addHandler(socketHandler)

    # spawn workers

    ctx = MultiprocessContext(
        name=f"{hostname}_",
        entrypoint=entrypoint,
        args={
            i: (
                WorkerArgs(
                    function=launcher_payload.fn,
                    master_hostname=main_agent_payload.hostname,
                    master_port=main_agent_payload.port,
                    backend=launcher_payload.backend,
                    rank=worker_global_ranks[i],
                    local_rank=i,
                    local_world_size=num_workers,
                    world_size=worker_world_size,
                    hostname=launcher_payload.hostnames[agent_rank],
                    log_host=launcher_payload.log_host,
                    log_port=launcher_payload.log_port,
                    timeout=launcher_payload.timeout,
                ).to_bytes(),
            )
            for i in range(num_workers)
        },
        envs={i: {} for i in range(num_workers)},
        logs_specs=DefaultLogsSpecs(log_dir=tempfile.mkdtemp()),
        start_method="spawn",
    )
    logger.debug("starting processes")
    try:
        ctx.start()

        status = AgentStatus()
        while True:
            if status.is_running():
                status = AgentStatus.from_result(
                    result=ctx.wait(5), worker_global_ranks=worker_global_ranks
                )

            agent_statuses = launcher_agent_group.sync_agent_statuses(status=status)

            if all(s.is_done() for s in agent_statuses):
                break

            if any(s.is_failed() for s in agent_statuses):
                raise RuntimeError("worker failure")

    except Exception as e:
        logger.error(f"encountered error: {e}")
        raise
    finally:
        ctx.close()

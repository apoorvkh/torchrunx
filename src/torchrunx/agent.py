from __future__ import annotations

__all__ = ["main"]

import datetime
import logging
import os
import socket
import sys
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Literal

import cloudpickle
import torch
import torch.distributed as dist
import torch.distributed.elastic.multiprocessing as dist_mp

from .logging_utils import log_records_to_socket, redirect_stdio_to_logger
from .utils import (
    AgentPayload,
    AgentStatus,
    ExceptionFromWorker,
    LauncherAgentGroup,
    get_open_port,
)


@dataclass
class WorkerArgs:
    function: Callable
    logger_hostname: str
    logger_port: int
    main_agent_hostname: str
    main_agent_port: int
    backend: Literal["nccl", "gloo", "mpi", "ucc", "auto"] | None
    rank: int
    local_rank: int
    local_world_size: int
    world_size: int
    hostname: str
    timeout: int

    def serialize(self) -> SerializedWorkerArgs:
        return SerializedWorkerArgs(worker_args=self)


class SerializedWorkerArgs:
    def __init__(self, worker_args: WorkerArgs) -> None:
        self.bytes = cloudpickle.dumps(worker_args)

    def deserialize(self) -> WorkerArgs:
        return cloudpickle.loads(self.bytes)


def _entrypoint(serialized_worker_args: SerializedWorkerArgs) -> Any | ExceptionFromWorker:
    worker_args: WorkerArgs = serialized_worker_args.deserialize()

    logger = logging.getLogger()

    log_records_to_socket(
        logger=logger,
        hostname=worker_args.hostname,
        local_rank=worker_args.local_rank,
        logger_hostname=worker_args.logger_hostname,
        logger_port=worker_args.logger_port,
    )

    redirect_stdio_to_logger(logger)

    os.environ["RANK"] = str(worker_args.rank)
    os.environ["LOCAL_RANK"] = str(worker_args.local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(worker_args.local_world_size)
    os.environ["WORLD_SIZE"] = str(worker_args.world_size)
    os.environ["MASTER_ADDR"] = worker_args.main_agent_hostname
    os.environ["MASTER_PORT"] = str(worker_args.main_agent_port)

    if worker_args.backend is not None:
        backend = worker_args.backend
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        dist.init_process_group(
            backend=backend,
            world_size=worker_args.world_size,
            rank=worker_args.rank,
            store=dist.TCPStore(  # pyright: ignore [reportPrivateImportUsage]
                host_name=worker_args.main_agent_hostname,
                port=worker_args.main_agent_port,
                world_size=worker_args.world_size,
                is_master=(worker_args.rank == 0),
            ),
            timeout=datetime.timedelta(seconds=worker_args.timeout),
        )

    try:
        return worker_args.function()
    except Exception as e:
        traceback.print_exc()
        return ExceptionFromWorker(exception=e)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def main(launcher_agent_group: LauncherAgentGroup, logger_hostname: str, logger_port: int) -> None:
    agent_rank = launcher_agent_group.rank - 1

    payload = AgentPayload(
        hostname=socket.getfqdn(),
        port=get_open_port(),
        process_id=os.getpid(),
    )

    launcher_payload, agent_payloads = launcher_agent_group.sync_payloads(payload=payload)
    main_agent_payload = agent_payloads[0]

    hostname = launcher_payload.hostnames[agent_rank]
    worker_world_size = launcher_payload.worker_world_size
    worker_global_ranks = launcher_payload.worker_global_ranks[agent_rank]
    num_workers = len(worker_global_ranks)

    logger = logging.getLogger()

    log_records_to_socket(
        logger=logger,
        hostname=hostname,
        local_rank=None,
        logger_hostname=logger_hostname,
        logger_port=logger_port,
    )

    redirect_stdio_to_logger(logger)

    # spawn workers

    ctx = dist_mp.start_processes(
        name=f"{hostname}_",
        entrypoint=_entrypoint,
        args={
            i: (
                WorkerArgs(
                    function=launcher_payload.fn,
                    logger_hostname=logger_hostname,
                    logger_port=logger_port,
                    main_agent_hostname=main_agent_payload.hostname,
                    main_agent_port=main_agent_payload.port,
                    backend=launcher_payload.backend,
                    rank=worker_global_ranks[i],
                    local_rank=i,
                    local_world_size=num_workers,
                    world_size=worker_world_size,
                    hostname=launcher_payload.hostnames[agent_rank],
                    timeout=launcher_payload.timeout,
                ).serialize(),
            )
            for i in range(num_workers)
        },
        # environment variables from agent are already automatically copied to workers
        envs={i: {} for i in range(num_workers)},
        # we handle logging ourselves, so we can discard these
        **(
            {"logs_specs": dist_mp.DefaultLogsSpecs(log_dir=tempfile.mkdtemp())}
            if torch.__version__ >= "2.3"
            else {"log_dir": tempfile.mkdtemp()}
        ),  # pyright: ignore [reportArgumentType]
    )

    try:
        status = None
        while True:
            if status is None or status.state == "running":
                # status can contain ExceptionFromWorker or WorkerFailedError
                status = AgentStatus.from_result(result=ctx.wait(5))

            # can raise AgentFailedError in launcher and all agents
            agent_statuses = launcher_agent_group.sync_agent_statuses(status=status)

            all_done = all(s.state == "done" for s in agent_statuses)
            any_failed = any(s.state == "failed" for s in agent_statuses)
            if all_done or any_failed:
                break
    finally:
        ctx.close()
        sys.stdout.flush()
        sys.stderr.flush()

from __future__ import annotations

import datetime
import logging
import os
import socket
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Literal

import cloudpickle
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing import start_processes
from typing_extensions import Self

from .logging_utils import log_records_to_socket, redirect_stdio_to_logger
from .utils import (
    AgentPayload,
    AgentStatus,
    LauncherAgentGroup,
    WorkerException,
    get_open_port,
)


@dataclass
class WorkerArgs:
    function: Callable
    logger_hostname: str
    logger_port: int
    main_agent_hostname: str
    main_agent_port: int
    backend: Literal["mpi", "gloo", "nccl", "ucc", None]
    rank: int
    local_rank: int
    local_world_size: int
    world_size: int
    hostname: str
    timeout: int

    def to_bytes(self) -> bytes:
        return cloudpickle.dumps(self)

    @classmethod
    def from_bytes(cls, serialized: bytes) -> Self:
        return cloudpickle.loads(serialized)


def entrypoint(serialized_worker_args: bytes) -> Any | WorkerException:
    worker_args = WorkerArgs.from_bytes(serialized_worker_args)

    logger = logging.getLogger()

    log_records_to_socket(
        logger=logger,
        hostname=worker_args.hostname,
        worker_rank=worker_args.local_rank,
        logger_hostname=worker_args.logger_hostname,
        logger_port=worker_args.logger_port,
    )

    redirect_stdio_to_logger(logger)

    store = dist.TCPStore(  # pyright: ignore[reportPrivateImportUsage]
        host_name=worker_args.main_agent_hostname,
        port=worker_args.main_agent_port,
        world_size=worker_args.world_size,
        is_master=(worker_args.rank == 0),
    )

    backend = worker_args.backend
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    logger.debug(f"using backend: {backend}")

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
    os.environ["MASTER_ADDR"] = worker_args.main_agent_hostname
    os.environ["MASTER_PORT"] = str(worker_args.main_agent_port)

    logger.debug(f"executing function: {worker_args.function}")

    try:
        return worker_args.function()
    except Exception as e:
        logger.error(e)
        return WorkerException(exception=e)
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
        worker_rank=None,
        logger_hostname=logger_hostname,
        logger_port=logger_port,
    )

    redirect_stdio_to_logger(logger)

    if torch.__version__ >= "2.3":
        from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs

        log_kwargs = {"logs_specs": DefaultLogsSpecs(log_dir=tempfile.mkdtemp())}
    else:
        log_kwargs = {"log_dir": tempfile.mkdtemp()}

    # spawn workers

    ctx = start_processes(
        name=f"{hostname}_",
        entrypoint=entrypoint,
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
                ).to_bytes(),
            )
            for i in range(num_workers)
        },
        envs={i: {} for i in range(num_workers)},
        **log_kwargs,  # pyright: ignore [reportArgumentType]
    )
    logger.info("starting processes")

    try:
        status = None
        while True:
            if status is None or status.state == "running":
                status = AgentStatus.from_result(
                    result=ctx.wait(5), worker_global_ranks=worker_global_ranks
                )

            agent_statuses = launcher_agent_group.sync_agent_statuses(status=status)

            if all(s.state == "done" for s in agent_statuses):
                break
            elif any(s.state == "failed" for s in agent_statuses):
                break
    except:
        raise
    finally:
        ctx.close()
        sys.stdout.flush()
        sys.stderr.flush()

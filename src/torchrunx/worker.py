"""Arguments and entrypoint for the worker processes."""

from __future__ import annotations

import datetime
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Literal

import cloudpickle
import torch
import torch.distributed as dist

from .utils.errors import ExceptionFromWorker
from .utils.logging import log_records_to_socket, redirect_stdio_to_logger

__all__ = ["WorkerArgs", "worker_entrypoint"]


@dataclass
class WorkerArgs:
    """Arguments passed from agent to spawned workers."""

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
        """Arguments must be serialized (to bytes) before passed to spawned workers."""
        return SerializedWorkerArgs(worker_args=self)


class SerializedWorkerArgs:
    """We use cloudpickle as a serialization backend (as it supports nearly all Python types)."""

    def __init__(self, worker_args: WorkerArgs) -> None:
        self.bytes = cloudpickle.dumps(worker_args)

    def deserialize(self) -> WorkerArgs:
        return cloudpickle.loads(self.bytes)


def worker_entrypoint(serialized_worker_args: SerializedWorkerArgs) -> Any | ExceptionFromWorker:
    """Function called by spawned worker processes.

    Workers first prepare a process group (for communicating with all other workers).
    They then invoke the user-provided function.
    Logs are transmitted to the launcher process.
    """
    worker_args: WorkerArgs = serialized_worker_args.deserialize()

    # Start logging to the logging server (i.e. the launcher)

    logger = logging.getLogger()

    log_records_to_socket(
        logger=logger,
        hostname=worker_args.hostname,
        local_rank=worker_args.local_rank,
        logger_hostname=worker_args.logger_hostname,
        logger_port=worker_args.logger_port,
    )

    redirect_stdio_to_logger(logger)

    # Set rank/world environment variables

    os.environ["RANK"] = str(worker_args.rank)
    os.environ["LOCAL_RANK"] = str(worker_args.local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(worker_args.local_world_size)
    os.environ["WORLD_SIZE"] = str(worker_args.world_size)
    os.environ["MASTER_ADDR"] = worker_args.main_agent_hostname
    os.environ["MASTER_PORT"] = str(worker_args.main_agent_port)

    # Prepare the process group (e.g. for communication within the user's function)

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

    # Invoke the user's function on this worker

    try:
        return worker_args.function()
    except Exception as e:
        traceback.print_exc()
        return ExceptionFromWorker(exception=e)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

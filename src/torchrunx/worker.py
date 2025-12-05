"""Arguments and entrypoint for the worker processes."""

from __future__ import annotations

import datetime
import logging
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import cloudpickle
import torch.distributed as dist

from .utils.errors import ExceptionFromWorker
from .utils.log_streaming import log_records_to_socket, redirect_stdio_to_logger

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["WorkerArgs", "worker_entrypoint"]


@dataclass
class WorkerArgs:
    """Arguments passed from agent to spawned workers."""

    function: Callable
    logger_hostname: str
    logger_port: int
    master_hostname: str
    master_port: int
    backend: Literal["nccl", "gloo", "mpi", "ucc"] | None
    rank: int
    local_rank: int
    node_rank: int
    local_world_size: int
    world_size: int
    hostname: str
    timeout: int

    def serialize(self) -> bytes:
        """Arguments must be serialized (to bytes) before passed to spawned workers."""
        return cloudpickle.dumps(asdict(self))

    @classmethod
    def from_bytes(cls, b: bytes) -> WorkerArgs:
        """Deserialize the bytes back into a WorkerArgs object."""
        return cls(**cloudpickle.loads(b))


def worker_entrypoint(serialized_worker_args: bytes) -> object | ExceptionFromWorker:
    """Function called by spawned worker processes.

    Workers first prepare a process group (for communicating with all other workers).
    They then invoke the user-provided function.
    Logs are transmitted to the launcher process.
    """
    worker_args = WorkerArgs.from_bytes(serialized_worker_args)

    # Start logging to the logging server (i.e. the launcher)

    log_records_to_socket(
        hostname=worker_args.hostname,
        local_rank=worker_args.local_rank,
        logger_hostname=worker_args.logger_hostname,
        logger_port=worker_args.logger_port,
    )

    logger = logging.getLogger()
    redirect_stdio_to_logger(logger)

    # Set rank/world environment variables

    os.environ["RANK"] = str(worker_args.rank)
    os.environ["LOCAL_RANK"] = str(worker_args.local_rank)
    os.environ["GROUP_RANK"] = str(worker_args.node_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(worker_args.local_world_size)
    os.environ["WORLD_SIZE"] = str(worker_args.world_size)
    os.environ["MASTER_ADDR"] = worker_args.master_hostname
    os.environ["MASTER_PORT"] = str(worker_args.master_port)

    # Prepare the process group (e.g. for communication within the user's function)

    if worker_args.backend is not None:
        backend = worker_args.backend

        dist.init_process_group(
            backend=backend,
            world_size=worker_args.world_size,
            rank=worker_args.rank,
            store=dist.TCPStore(  # pyright: ignore [reportPrivateImportUsage]
                host_name=worker_args.master_hostname,
                port=worker_args.master_port,
                world_size=worker_args.world_size,
                is_master=(worker_args.rank == 0),
            ),
            timeout=datetime.timedelta(seconds=worker_args.timeout),
        )

    # Invoke the user's function on this worker

    try:
        return worker_args.function()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return ExceptionFromWorker(exception=e)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

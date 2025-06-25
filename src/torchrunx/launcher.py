"""For launching functions with our library."""

from __future__ import annotations

__all__ = ["DEFAULT_ENV_VARS_FOR_COPY", "LaunchResult", "Launcher"]

import fnmatch
import itertools
import logging
import os
import socket
import typing
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Event, Process
from typing import Generic, TypeVar

import torch.distributed as dist
from typing_extensions import ParamSpec, Self

from .utils.comm import (
    LauncherAgentGroup,
    LauncherPayload,
    get_open_port,
)
from .utils.environment import (
    build_launch_command,
    execute_command,
    resolve_environment,
)
from .utils.errors import ExceptionFromWorker, WorkerFailedError
from .utils.log_handling import default_handlers
from .utils.log_streaming import LoggingServerArgs, start_logging_server

DEFAULT_ENV_VARS_FOR_COPY = (
    "PATH",
    "LD_LIBRARY",
    "LIBRARY_PATH",
    "PYTHON*",
    "CUDA*",
    "TORCH*",
    "PYTORCH*",
    "NCCL*",
)

FunctionP = ParamSpec("FunctionP")
FunctionR = TypeVar("FunctionR")


@dataclass
class Launcher:
    """For configuring the function launch environment."""

    hostnames: list[str] | typing.Literal["auto", "slurm"] = "auto"
    """Nodes to launch the function on. By default, infer from SLURM, else ``["localhost"]``."""
    workers_per_host: int | list[int] | typing.Literal["cpu", "gpu"] = "gpu"
    """Number of processes to run per node. By default, number of GPUs per host."""
    ssh_config_file: str | os.PathLike | None = None
    """For connecting to nodes. By default, ``"~/.ssh/config"`` or ``"/etc/ssh/ssh_config"``."""
    backend: typing.Literal["nccl", "gloo", "mpi", "ucc"] | None = "nccl"
    """`Backend <https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend>`_
        for worker process group. By default, NCCL (GPU backend).
        Use GLOO for CPU backend. ``None`` for no process group."""
    worker_timeout: int = 600
    """Worker process group timeout (seconds)."""
    agent_timeout: int = 180
    """Agent communication timeout (seconds)."""
    copy_env_vars: tuple[str, ...] = DEFAULT_ENV_VARS_FOR_COPY
    """Environment variables to copy from the launcher process to workers.
       Supports Unix pattern matching syntax."""
    extra_env_vars: dict[str, str] | None = None
    """Additional environment variables to load onto workers."""
    env_file: str | os.PathLike | None = None
    """Path to a ``.env`` file, containing environment variables to load onto workers."""

    handler_factory: typing.Callable[[], list[logging.Handler]] | typing.Literal["auto"] | None = (
        field(default="auto", init=False)
    )

    def set_logging_handlers(
        self,
        handler_factory: typing.Callable[[], list[logging.Handler]] | typing.Literal["auto"] | None,
    ) -> Self:
        """Provide a ``handler_factory`` function to customize processing of agent/worker logs.

        Parameters:
          handler_factory: Function that constructs and returns :obj:`logging.Handler` objects.
              See `Custom Logging <https://torchrun.xyz/usage/logging.html>`_ for more details.
        """
        self.handler_factory = handler_factory
        return self

    def run(  # noqa: C901, PLR0912, PLR0915
        self,
        func: typing.Callable[FunctionP, FunctionR],
        *args: FunctionP.args,
        **kwargs: FunctionP.kwargs,
    ) -> LaunchResult[FunctionR]:
        """Distribute a function onto specified nodes and parallelize across workers.

        Raises:
            RuntimeError: Configuration issues.
            Exception: Exceptions raised in worker processes are propagated.
            WorkerFailedError: If a worker fails (e.g. from a segmentation fault).
            AgentFailedError: If an agent fails, e.g. from an OS signal.
        """
        logger = logging.getLogger(__package__)

        if not dist.is_available():
            msg = "The torch.distributed package is not available."
            raise RuntimeError(msg)

        logger.debug("Preparing launch environment.")

        ###

        hostnames, workers_per_host = resolve_environment(
            self.hostnames, self.workers_per_host, ssh_config_file=self.ssh_config_file
        )
        ssh_config_file = self.ssh_config_file
        backend = self.backend
        worker_timeout = self.worker_timeout
        agent_timeout = self.agent_timeout

        env_vars = {
            k: v
            for k, v in os.environ.items()
            if any(fnmatch.fnmatch(k, e) for e in self.copy_env_vars)
        }
        if self.extra_env_vars is not None:
            env_vars.update(self.extra_env_vars)
        env_file = self.env_file

        if self.handler_factory is None:

            def handler_factory() -> list[logging.Handler]:
                return []
        elif self.handler_factory == "auto":
            handler_factory = partial(default_handlers, hostnames, workers_per_host)
        else:
            handler_factory = self.handler_factory

        ###

        launcher_hostname = socket.getfqdn()
        launcher_port = get_open_port()
        logging_port = get_open_port()
        world_size = len(hostnames) + 1

        stop_logging_event = None
        log_process = None
        launcher_agent_group = None

        _cumulative_workers = [0, *itertools.accumulate(workers_per_host)]
        worker_global_ranks = [
            list(range(_cumulative_workers[n], _cumulative_workers[n + 1]))
            for n in range(len(hostnames))
        ]
        payload = LauncherPayload(
            fn=partial(func, *args, **kwargs),
            hostnames=hostnames,
            worker_global_ranks=worker_global_ranks,
            worker_world_size=sum(workers_per_host),
            backend=backend,
            worker_timeout=worker_timeout,
        )
        agent_payloads = None

        try:
            logger.debug("Starting logging server.")

            # Start logging server (recieves LogRecords from agents/workers)

            logging_server_args = LoggingServerArgs(
                handler_factory=handler_factory,
                logging_hostname=launcher_hostname,
                logging_port=logging_port,
            )

            stop_logging_event = Event()

            log_process = Process(
                target=start_logging_server,
                args=(logging_server_args.serialize(), stop_logging_event),
                daemon=True,
            )

            log_process.start()

            # Start agents on each node

            for i, hostname in enumerate(hostnames):
                logger.info(f'Launching "{func.__name__}" on {hostname}.')

                execute_command(
                    command=build_launch_command(
                        launcher_hostname=launcher_hostname,
                        launcher_port=launcher_port,
                        logger_port=logging_port,
                        world_size=world_size,
                        rank=i + 1,
                        env_vars=env_vars,
                        env_file=env_file,
                        hostname=hostname,
                        agent_timeout=agent_timeout,
                    ),
                    hostname=hostname,
                    ssh_config_file=ssh_config_file,
                )

            logger.debug("Initializing launcher-agent group.")

            # Initialize launcher-agent process group
            # ranks = (launcher, agent_{hostnames[0]}, ..., agent[-1])

            launcher_agent_group = LauncherAgentGroup[FunctionR](
                launcher_hostname=launcher_hostname,
                launcher_port=launcher_port,
                world_size=world_size,
                rank=0,
                agent_timeout=agent_timeout,
            )

            # Sync initial payloads between launcher and agents

            logger.debug("Synchronizing launcher and agents.")
            launcher_payload, agent_payloads = launcher_agent_group.sync_payloads(payload=payload)

            # Monitor agent statuses (until failed or done)

            logger.debug("Entering agent monitoring loop.")

            while True:
                # could raise AgentFailedError
                agent_statuses = launcher_agent_group.sync_agent_statuses(status=None)

                # raises specific exception if any agent fails
                for s in agent_statuses:
                    for v in s.return_values:
                        if isinstance(v, ExceptionFromWorker):
                            raise v.exception
                        if isinstance(v, WorkerFailedError):
                            raise v

                if all(s.state == "done" for s in agent_statuses):
                    logger.info("All workers completed successfully.")
                    return_values: list[list[FunctionR]] = [s.return_values for s in agent_statuses]  # pyright: ignore [reportAssignmentType]
                    return LaunchResult.from_returns(hostnames, return_values)
        finally:
            # cleanup: SIGTERM all agents
            if agent_payloads is not None:
                for agent_payload, agent_hostname in zip(agent_payloads, hostnames):
                    logger.debug("Killing PID %s on %s.", agent_payload.process_id, agent_hostname)

                    execute_command(
                        command=f"kill {agent_payload.process_id}",
                        hostname=agent_hostname,
                        ssh_config_file=ssh_config_file,
                    )

            if launcher_agent_group is not None:
                logger.debug("Killing launcher-agent group.")
                launcher_agent_group.shutdown()

            logger.debug("Stopping logging server.")

            if stop_logging_event is not None:
                stop_logging_event.set()
            if log_process is not None:
                log_process.kill()


@dataclass
class LaunchResult(Generic[FunctionR]):
    """Container for objects returned from workers after successful launches."""

    results: dict[str, list[FunctionR]]  # [hostname][local_rank] -> FunctionR

    @classmethod
    def from_returns(cls, hostnames: list[str], return_values: list[list[FunctionR]]) -> Self:  # noqa: D102
        return cls(results=dict(zip(hostnames, return_values)))

    def index(self, hostname: str, locak_rank: int) -> FunctionR:
        """Get return value from worker by host and local rank."""
        return self.results[hostname][locak_rank]

    def rank(self, i: int) -> FunctionR:
        """Get return value from worker by global rank."""
        for results_per_host in self.results.values():
            if i < len(results_per_host):
                return results_per_host[i]
            i -= len(results_per_host)
        raise IndexError

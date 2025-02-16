"""For launching functions with our library."""

from __future__ import annotations

__all__ = ["LaunchResult", "Launcher"]

import itertools
import socket
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Event, Process
from typing import TYPE_CHECKING, Generic, TypeVar

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
from .utils.logging import LoggingServerArgs, start_logging_server

if TYPE_CHECKING:
    import logging
    import os
    import typing


FunctionP = ParamSpec("FunctionP")
FunctionR = TypeVar("FunctionR")


@dataclass
class Launcher:
    """For configuring the function launch environment."""

    hostnames: list[str] | typing.Literal["auto", "slurm"] = "auto"
    """Nodes on which to launch the function. By default, infer from localhost or SLURM."""
    workers_per_host: int | list[int] | typing.Literal["auto"] = "auto"
    """Number of processes to run per node. By default, number of GPUs per host."""
    ssh_config_file: str | os.PathLike | None = None
    """For connecting to nodes. By default, ``"~/.ssh/config"`` or ``"/etc/ssh/ssh_config"``."""
    backend: typing.Literal["nccl", "gloo", "mpi", "ucc", "auto"] | None = "auto"
    """`Backend <https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend>`_
        for worker process group or ``None``. By default, NCCL if GPUs detected, else GLOO."""
    timeout: int = 600
    """Worker process group timeout (seconds)."""
    default_env_vars: tuple[str, ...] = (
        "PATH",
        "LD_LIBRARY",
        "LIBRARY_PATH",
        "PYTHON*",
        "CUDA*",
        "TORCH*",
        "PYTORCH*",
        "NCCL*",
    )
    """Environment variables to copy from the launcher process to workers.
       Supports bash pattern matching syntax."""
    extra_env_vars: tuple[str, ...] = ()
    """Additional user-specified environment variables to copy."""
    env_file: str | os.PathLike | None = None
    """Path to (e.g. ``.env``) with additional environment variables to load onto workers."""
    propagate_exceptions: bool = True
    """Whether to raise specific worker exceptions or :exc:`torchrunx.WorkerFailedError`."""

    handler_factory: typing.Callable[[], list[logging.Handler]] | typing.Literal["auto"] | None = (
        field(default="auto", init=False)
    )

    def set_handler_factory(
        self, factory: typing.Callable[[], list[logging.Handler]] | typing.Literal["auto"] | None
    ) -> Self:
        """Provide a ``factory`` to set custom handling of agent and worker logs.

        Parameters:
          factory: Factory function to generate :obj:`logging.Handler` objects.

        See `custom logging <https://torchrun.xyz/features/customization.html#logging>`_.
        """
        self.handler_factory = factory
        return self

    def run(  # noqa: C901, PLR0912
        self,
        func: typing.Callable[FunctionP, FunctionR],
        *args: FunctionP.args,
        **kwargs: FunctionP.kwargs,
    ) -> LaunchResult[FunctionR]:
        """Distribute a function onto specified nodes and parallelize across workers.

        Raises:
            RuntimeError: Configuration issues.
            Exception: Exceptions raised in worker processes are propagated
                (if ``propagate_exceptions=True``).
            WorkerFailedError: If a worker fails (e.g. from a segmentation fault)
                or raises an exception with ``propagate_exceptions=False``.
            AgentFailedError: If an agent fails, e.g. from an OS signal.
        """
        if not dist.is_available():
            msg = "The torch.distributed package is not available."
            raise RuntimeError(msg)

        hostnames, workers_per_host, backend = resolve_environment(
            self.hostnames, self.workers_per_host, self.backend, self.ssh_config_file
        )

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
            fn=partial(func, *(args or ()), **(kwargs or {})),
            hostnames=hostnames,
            worker_global_ranks=worker_global_ranks,
            worker_world_size=sum(workers_per_host),
            backend=backend,
            timeout=self.timeout,
        )
        agent_payloads = None

        try:
            # Start logging server (recieves LogRecords from agents/workers)

            logging_server_args = LoggingServerArgs(
                handler_factory=self.handler_factory,
                logging_hostname=launcher_hostname,
                logging_port=logging_port,
                hostnames=hostnames,
                workers_per_host=workers_per_host,
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
                execute_command(
                    command=build_launch_command(
                        launcher_hostname=launcher_hostname,
                        launcher_port=launcher_port,
                        logger_port=logging_port,
                        world_size=world_size,
                        rank=i + 1,
                        env_vars=(self.default_env_vars + self.extra_env_vars),
                        env_file=self.env_file,
                    ),
                    hostname=hostname,
                    ssh_config_file=self.ssh_config_file,
                )

            # Initialize launcher-agent process group
            # ranks = (launcher, agent_{hostnames[0]}, ..., agent[-1])

            launcher_agent_group = LauncherAgentGroup[FunctionR](
                launcher_hostname=launcher_hostname,
                launcher_port=launcher_port,
                world_size=world_size,
                rank=0,
            )

            # Sync initial payloads between launcher and agents

            launcher_payload, agent_payloads = launcher_agent_group.sync_payloads(payload=payload)

            # Monitor agent statuses (until failed or done)

            while True:
                # could raise AgentFailedError
                agent_statuses = launcher_agent_group.sync_agent_statuses(status=None)

                # raises specific exception if any agent fails
                for s in agent_statuses:
                    for v in s.return_values:
                        if isinstance(v, ExceptionFromWorker):
                            if self.propagate_exceptions:
                                raise v.exception
                            raise WorkerFailedError from v.exception
                        if isinstance(v, WorkerFailedError):
                            raise v

                if all(s.state == "done" for s in agent_statuses):
                    return_values: list[list[FunctionR]] = [s.return_values for s in agent_statuses]  # pyright: ignore [reportAssignmentType]
                    return LaunchResult.from_returns(hostnames, return_values)
        finally:
            if stop_logging_event is not None:
                stop_logging_event.set()
            if log_process is not None:
                log_process.kill()

            if launcher_agent_group is not None:
                launcher_agent_group.shutdown()

            # cleanup: SIGTERM all agents
            if agent_payloads is not None:
                for agent_payload, agent_hostname in zip(agent_payloads, hostnames):
                    execute_command(
                        command=f"kill {agent_payload.process_id}",
                        hostname=agent_hostname,
                        ssh_config_file=self.ssh_config_file,
                    )


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

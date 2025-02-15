"""For launching functions with our library."""

from __future__ import annotations

__all__ = ["LaunchResult", "Launcher"]

import itertools
import socket
from dataclasses import dataclass
from functools import partial
from multiprocessing import Event, Process
from typing import TYPE_CHECKING, Callable, Generic, Literal, TypeVar

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
    resolve_hostnames,
    resolve_workers_per_host,
)
from .utils.errors import ExceptionFromWorker, WorkerFailedError
from .utils.logging import LoggingServerArgs, start_logging_server

if TYPE_CHECKING:
    import os
    from logging import Handler


"""Distribute and parallelize a function onto specified nodes and workers.

Arguments:
    func: Function to replicate on each node/worker.
    args: Positional arguments for ``func``. Default: :py:obj:`None`.
    kwargs: Keyword arguments for ``func``. Default: :py:obj:`None`.
    hostnames: Nodes on which to launch the function.
        Default: ``"auto"`` (infer from localhost or SLURM).
    workers_per_host: Number of processes to run (e.g. # of GPUs) per node.
        Default: ``"auto"`` (number of GPUs per host).
    ssh_config_file: Path to an SSH configuration file for connecting to nodes.
        Default: ``"~/.ssh/config"`` or ``"/etc/ssh/ssh_config"``.
    backend: `Backend <https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend>`_
        for worker process group. Set `None` to disable.
        Default: ``"auto"`` (NCCL if GPU or GLOO if CPU).
    timeout: Worker process group timeout (seconds).
        Default: ``600``.
    default_env_vars: Environment variables to copy from the launcher process to workers.
        Supports bash pattern matching syntax.
        Default: ``("PATH", "LD_LIBRARY", "LIBRARY_PATH", "PYTHON*", "CUDA*", "TORCH*",
        "PYTORCH*", "NCCL*")``.
    extra_env_vars: Additional user-specified environment variables to copy.
        Default: ``()``.
    env_file: Path to a file (e.g., ``.env``) with additional environment variables to copy.
        Default: :py:obj:`None`.
    propagate_exceptions: Raise exceptions from worker processes in the launcher.
        If false, raises :exc:`WorkerFailedError` instead.
        Default: :py:obj:`True`.
    handler_factory: Function to customize processing of agent and worker logs with handlers.
        Default: ``"auto"`` (see `custom logging <https://torchrun.xyz/features/customization.html#logging>`_).

Raises:
    RuntimeError: If there are configuration issues.
    Exception: Any exception raised in a worker process is propagated.
    WorkerFailedError: If a worker fails (e.g. from a segmentation fault)
        or raises an exception and ``propagate_exceptions=False``.
    AgentFailedError: If an agent fails, e.g. from an OS signal.
"""


FunctionP = ParamSpec("FunctionP")
FunctionR = TypeVar("FunctionR")


@dataclass
class Launcher:
    """Alias class for :func:`launch`. Refer to that function for documentation."""

    hostnames: list[str] | Literal["auto", "slurm"] = "auto"
    """Node hostnames to use in distributed execution. "auto" and "slurm" attempt to detect this
    for you based on your environmental variables."""
    workers_per_host: int | list[int] | Literal["auto"] = "auto"
    """Number of worker processes per node. You can specify a constant number of workers for all
    nodes (int), a different number of workers for each node (list[int]), or automatically determine
    it per-node ("auto")."""
    ssh_config_file: str | os.PathLike | None = None
    """Path to custom SSH Config for passwordless SSH into each node."""
    backend: Literal["nccl", "gloo", "mpi", "ucc", "auto"] | None = "auto"
    """A torch.distributed backend to use for inter-process communication. "auto" will use NCCL if
    GPUs are detected, otherwise GLOO."""
    timeout: int = 600
    """The torch.distributed communication timeout of the worker process group, in seconds."""
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
    """Environmental variables to clone from the launcher process to worker processes,
    supporting unix pattern matching."""
    extra_env_vars: tuple[str, ...] = ()
    """Additional environmental variables to set in the worker process environments,
    formatted identically to the defaul_env_vars field."""
    env_file: str | os.PathLike | None = None
    """A bash style .env file that will be sourced by worker processes."""
    propagate_exceptions: bool = True
    """Whether worker exceptions should be raised by the launcher."""

    def __post_init__(self) -> None:
        """Initializing ``handler_factory``. Inclusion in ``__init__`` inhibits CLI generation."""
        self.handler_factory: Callable[[], list[Handler]] | Literal["auto"] | None = "auto"

    def set_handler_factory(
        self, factory: Callable[[], list[Handler]] | Literal["auto"] | None
    ) -> Self:
        """Setter for log handler factory."""
        self.handler_factory = factory
        return self

    def run(  # noqa: C901, PLR0912
        self,
        func: Callable[FunctionP, FunctionR],
        *args: FunctionP.args,
        **kwargs: FunctionP.kwargs,
    ) -> LaunchResult[FunctionR | WorkerFailedError | ExceptionFromWorker]:
        """Launch a function using class configuration."""
        if not dist.is_available():
            msg = "The torch.distributed package is not available."
            raise RuntimeError(msg)

        hostnames: list[str] = resolve_hostnames(self.hostnames)
        workers_per_host: list[int] = resolve_workers_per_host(hostnames, self.workers_per_host)

        launcher_hostname = socket.getfqdn()
        launcher_port = get_open_port()
        logging_port = get_open_port()
        world_size = len(hostnames) + 1

        stop_logging_event = None
        log_process = None
        launcher_agent_group = None
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
                backend=self.backend,
                timeout=self.timeout,
            )

            launcher_payload, agent_payloads = launcher_agent_group.sync_payloads(payload=payload)

            # Monitor agent statuses (until failed or done)

            while True:
                # could raise AgentFailedError
                agent_statuses = launcher_agent_group.sync_agent_statuses(status=None)

                # raises specific exception if any agent fails
                for s in agent_statuses:
                    for value in s.return_values:
                        if isinstance(value, ExceptionFromWorker):
                            if self.propagate_exceptions:
                                raise value.exception
                            raise WorkerFailedError from value.exception
                        if isinstance(value, WorkerFailedError):
                            raise value

                if all(s.state == "done" for s in agent_statuses):
                    break
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

        # if launch is successful: return objects from workers
        return_values = [s.return_values for s in agent_statuses]
        return LaunchResult(hostnames=hostnames, return_values=return_values)


class LaunchResult(Generic[FunctionR]):
    """Container for objects returned from workers after successful launches."""

    results: dict[str, list[FunctionR]]

    def __init__(self, hostnames: list[str], return_values: list[list[FunctionR]]) -> None:
        """Initialize from corresponding lists of hostnames and worker return values."""
        self.results: dict[str, list[FunctionR]] = dict(zip(hostnames, return_values))

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

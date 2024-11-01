"""For launching functions with our library."""

from __future__ import annotations

__all__ = ["Launcher", "launch", "LaunchResult"]

import fnmatch
import ipaddress
import itertools
import logging
import os
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from functools import partial, reduce
from logging import Handler
from multiprocessing import Event, Process
from operator import add
from pathlib import Path
from typing import Any, Callable, Literal

import fabric
import torch.distributed as dist

from .utils.comm import (
    LauncherAgentGroup,
    LauncherPayload,
    get_open_port,
)
from .utils.environment import auto_hosts, auto_workers, slurm_hosts, slurm_workers
from .utils.errors import (
    ExceptionFromWorker,
    WorkerFailedError,
)
from .utils.logging import LoggingServerArgs, start_logging_server


@dataclass
class Launcher:
    """Useful for sequential invocations or for specifying arguments via CLI."""

    hostnames: list[str] | Literal["auto", "slurm"] = "auto"
    workers_per_host: int | list[int] | Literal["auto", "slurm"] = "auto"
    ssh_config_file: str | os.PathLike | None = None
    backend: Literal["nccl", "gloo", "mpi", "ucc", "auto"] | None = "auto"
    timeout: int = 600
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
    extra_env_vars: tuple[str, ...] = ()
    env_file: str | os.PathLike | None = None

    def run(  # noqa: C901, PLR0912
        self,
        func: Callable,
        func_args: tuple[Any] | None = None,
        func_kwargs: dict[str, Any] | None = None,
        handler_factory: Callable[[], list[Handler]] | Literal["auto"] | None = "auto",
    ) -> LaunchResult:
        """Run a function using the :mod:`torchrunx.Launcher` configuration."""
        if not dist.is_available():
            msg = "The torch.distributed package is not available."
            raise RuntimeError(msg)

        hostnames = _resolve_hostnames(self.hostnames)
        workers_per_host = _resolve_workers_per_host(self.workers_per_host, len(hostnames))

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
                handler_factory=handler_factory,
                logging_hostname=launcher_hostname,
                logging_port=logging_port,
                hostnames=hostnames,
                workers_per_host=workers_per_host,
                log_dir=Path(os.environ.get("TORCHRUNX_LOG_DIR", "torchrunx_logs")),
                log_level=logging._nameToLevel[os.environ.get("TORCHRUNX_LOG_LEVEL", "INFO")],  # noqa: SLF001
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
                _execute_command(
                    command=_build_launch_command(
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

            launcher_agent_group = LauncherAgentGroup(
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
                fn=partial(func, *(func_args or ()), **(func_kwargs or {})),
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
                            raise value.exception
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
                    _execute_command(
                        command=f"kill {agent_payload.process_id}",
                        hostname=agent_hostname,
                        ssh_config_file=self.ssh_config_file,
                    )

        # if launch is successful: return objects from workers
        return_values = [s.return_values for s in agent_statuses]
        return LaunchResult(hostnames=hostnames, return_values=return_values)


def launch(
    func: Callable,
    func_args: tuple[Any] | None = None,
    func_kwargs: dict[str, Any] | None = None,
    hostnames: list[str] | Literal["auto", "slurm"] = "auto",
    workers_per_host: int | list[int] | Literal["auto", "slurm"] = "auto",
    ssh_config_file: str | os.PathLike | None = None,
    backend: Literal["nccl", "gloo", "mpi", "ucc", "auto"] | None = "auto",
    timeout: int = 600,
    default_env_vars: tuple[str, ...] = (
        "PATH",
        "LD_LIBRARY",
        "LIBRARY_PATH",
        "PYTHON*",
        "CUDA*",
        "TORCH*",
        "PYTORCH*",
        "NCCL*",
    ),
    extra_env_vars: tuple[str, ...] = (),
    env_file: str | os.PathLike | None = None,
    handler_factory: Callable[[], list[Handler]] | Literal["auto"] | None = "auto",
) -> LaunchResult:
    """Launch a distributed PyTorch function on the specified nodes.

    Arguments:
        func: Function to run on each worker.
        func_args: Positional arguments for ``func``.
        func_kwargs: Keyword arguments for ``func``.
        hostnames: Nodes on which to launch the function.
            Defaults to nodes inferred from a SLURM environment or localhost.
        workers_per_host: Number of processes to run per node.
            Can specify different counts per node with a list.
        ssh_config_file: Path to an SSH configuration file for connecting to nodes.
            Defaults to ``~/.ssh/config`` or ``/etc/ssh/ssh_config``.
        backend: `Backend <https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend>`_
            for worker process group. Defaults to NCCL (GPU) or GLOO (CPU). Set `None` to disable.
        timeout: Worker process group timeout (seconds).
        default_env_vars: Environment variables to copy from the launcher process to workers.
            Supports bash pattern matching syntax.
        extra_env_vars: Additional user-specified environment variables to copy.
        env_file: Path to a file (e.g., `.env`) with additional environment variables to copy.
        handler_factory: Function to build logging handlers that process agent and worker logs.
            Defaults to an automatic basic logging scheme.

    Raises:
        RuntimeError: If there are configuration issues.
        AgentFailedError: If an agent fails, e.g. from an OS signal.
        WorkerFailedError: If a worker fails, e.g. from a segmentation fault.
        Exception: Any exception raised in a worker process is propagated.
    """
    return Launcher(
        hostnames=hostnames,
        workers_per_host=workers_per_host,
        ssh_config_file=ssh_config_file,
        backend=backend,
        timeout=timeout,
        default_env_vars=default_env_vars,
        extra_env_vars=extra_env_vars,
        env_file=env_file,
    ).run(
        func=func,
        func_args=func_args,
        func_kwargs=func_kwargs,
        handler_factory=handler_factory,
    )


@dataclass
class LaunchResult:
    """Container for objects returned from workers after successful launches."""

    hostnames: list[str]
    return_values: list[list[Any]]

    def by_hostnames(self) -> dict[str, list[Any]]:
        """All return values from workers, indexed by host and local rank."""
        return dict(zip(self.hostnames, self.return_values))

    def by_ranks(self) -> list[Any]:
        """All return values from workers, indexed by global rank."""
        return reduce(add, self.return_values)

    def index(self, hostname: str, rank: int) -> Any:
        """Get return value from worker by host and local rank."""
        return self.return_values[self.hostnames.index(hostname)][rank]

    def rank(self, i: int) -> Any:
        """Get return value from worker by global rank."""
        return self.by_ranks()[i]


def _resolve_hostnames(hostnames: list[str] | Literal["auto", "slurm"]) -> list[str]:
    if hostnames == "auto":
        return auto_hosts()
    if hostnames == "slurm":
        return slurm_hosts()
    return hostnames


def _resolve_workers_per_host(
    workers_per_host: int | list[int] | Literal["auto", "slurm"],
    num_hosts: int,
) -> list[int]:
    if workers_per_host == "auto":
        workers_per_host = auto_workers()
    elif workers_per_host == "slurm":
        workers_per_host = slurm_workers()

    if isinstance(workers_per_host, int):
        workers_per_host = [workers_per_host] * num_hosts
    elif len(workers_per_host) != num_hosts:
        msg = "len(workers_per_host) != len(hostnames)"
        raise ValueError(msg)

    return workers_per_host


def _build_launch_command(
    launcher_hostname: str,
    launcher_port: int,
    logger_port: int,
    world_size: int,
    rank: int,
    env_vars: tuple[str, ...],
    env_file: str | os.PathLike | None,
) -> str:
    # shlex.quote prevents shell injection here (resolves S602 in execute_command)

    commands = []

    current_dir = shlex.quote(str(Path.cwd()))
    commands.append("cd " + current_dir)

    env_exports = []
    for k, v in os.environ.items():
        if any(fnmatch.fnmatch(k, e) for e in env_vars):
            env_exports.append(shlex.quote(f"{k}={v}"))

    if len(env_exports) > 0:
        commands.append("export " + " ".join(env_exports))

    if env_file is not None:
        commands.append("source " + shlex.quote(str(env_file)))

    python = shlex.quote(sys.executable)
    launcher_hostname = shlex.quote(launcher_hostname)

    commands.append(
        f"{python} -u -m torchrunx "
        f"--launcher-hostname {launcher_hostname} "
        f"--launcher-port {launcher_port} "
        f"--logger-port {logger_port} "
        f"--world-size {world_size} "
        f"--rank {rank}",
    )

    return " && ".join(commands)


def _execute_command(
    command: str,
    hostname: str,
    ssh_config_file: str | os.PathLike | None = None,
) -> None:
    is_localhost = True
    _hostname_or_ip = hostname
    try:
        _ip = ipaddress.ip_address(_hostname_or_ip)
    except ValueError:
        _ip = ipaddress.ip_address(socket.gethostbyname(_hostname_or_ip))
    if not _ip.is_loopback:
        # compare local interface addresses between host and localhost
        _host_addrs = [addr[4][0] for addr in socket.getaddrinfo(str(_ip), None)]
        _localhost_addrs = [addr[4][0] for addr in socket.getaddrinfo(socket.gethostname(), None)]
        is_localhost = len(set(_host_addrs) & set(_localhost_addrs)) > 0

    if is_localhost:
        # S602: subprocess.Popen is called with shell=True (https://docs.python.org/3.9/library/subprocess.html#security-considerations)
        # Made sure to shlex.quote arguments in build_command to prevent shell injection
        subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S602
    else:
        runtime_ssh_path = ssh_config_file
        if isinstance(ssh_config_file, os.PathLike):
            runtime_ssh_path = str(ssh_config_file)

        with fabric.Connection(
            host=hostname,
            config=fabric.Config(runtime_ssh_path=runtime_ssh_path),
        ) as conn:
            conn.run(f"{command} >> /dev/null 2>&1 &", asynchronous=True)

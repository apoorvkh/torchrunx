from __future__ import annotations

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
from multiprocessing import Process
from operator import add
from pathlib import Path
from typing import Any, Callable, Literal, overload

import fabric
import torch.distributed as dist

from .environment import auto_hosts, auto_workers, slurm_hosts, slurm_workers
from .logging_utils import LogRecordSocketReceiver, default_handlers
from .utils import AgentStatus, LauncherAgentGroup, LauncherPayload, WorkerException, get_open_port


def resolve_hostnames(hostnames: list[str] | Literal["auto", "slurm"]) -> list[str]:
    if hostnames == "auto":
        return auto_hosts()
    if hostnames == "slurm":
        return slurm_hosts()
    return hostnames


def resolve_workers_per_host(
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


def build_logging_server(
    log_handlers: list[Handler] | Literal["auto"] | None,
    launcher_hostname: str,
    hostnames: list[str],
    workers_per_host: list[int],
    log_dir: str | os.PathLike,
    log_level: int,
) -> LogRecordSocketReceiver:
    if log_handlers is None:
        log_handlers = []
    elif log_handlers == "auto":
        log_handlers = default_handlers(
            hostnames=hostnames,
            workers_per_host=workers_per_host,
            log_dir=log_dir,
            log_level=log_level,
        )

    return LogRecordSocketReceiver(
        host=launcher_hostname,
        port=get_open_port(),
        handlers=log_handlers,
    )


def build_launch_command(
    launcher_hostname: str,
    launcher_port: int,
    logger_port: int,
    world_size: int,
    rank: int,
    env_vars: list[str] | tuple[str],
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


def execute_command(
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
        # S602: subprocess.Popen is called with shell=True (https://docs.python.org/3.8/library/subprocess.html#security-considerations)
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


@dataclass
class Launcher:
    hostnames: list[str] | Literal["auto", "slurm"] = "auto"
    workers_per_host: int | list[int] | Literal["auto", "slurm"] = "auto"
    ssh_config_file: str | os.PathLike | None = None
    backend: Literal["nccl", "gloo", "mpi", "ucc", "auto"] | None = "auto"
    timeout: int = 600
    log_handlers: list[Handler] | Literal["auto"] | None = "auto"
    default_env_vars: tuple[str] = (  # pyright: ignore [reportAssignmentType]
        "PATH",
        "LD_LIBRARY",
        "LIBRARY_PATH",
        "PYTHON*",
        "CUDA*",
        "TORCH*",
        "PYTORCH*",
        "NCCL*",
    )
    extra_env_vars: tuple[str] = ()
    env_file: str | os.PathLike | None = None

    def run(  # noqa: C901, PLR0912
        self,
        func: Callable,
        func_args: tuple[Any] | None = None,
        func_kwargs: dict[str, Any] | None = None,
    ) -> LaunchResult:
        if not dist.is_available():
            msg = "The torch.distributed package is not available."
            raise RuntimeError(msg)

        hostnames = resolve_hostnames(self.hostnames)
        workers_per_host = resolve_workers_per_host(self.workers_per_host, len(hostnames))

        launcher_hostname = socket.getfqdn()
        launcher_port = get_open_port()
        world_size = len(hostnames) + 1

        log_receiver = None
        log_process = None
        launcher_agent_group = None
        agent_payloads = None

        try:
            # start logging server

            log_receiver = build_logging_server(
                log_handlers=self.log_handlers,
                launcher_hostname=launcher_hostname,
                hostnames=hostnames,
                workers_per_host=workers_per_host,
                log_dir=Path(os.environ.get("TORCHRUNX_LOG_DIR", "torchrunx_logs")),
                log_level=logging._nameToLevel[os.environ.get("TORCHRUNX_LOG_LEVEL", "INFO")],  # noqa: SLF001
            )

            log_process = Process(
                target=log_receiver.serve_forever,
                daemon=True,
            )

            log_process.start()

            # start agents on each node

            for i, hostname in enumerate(hostnames):
                execute_command(
                    command=build_launch_command(
                        launcher_hostname=launcher_hostname,
                        launcher_port=launcher_port,
                        logger_port=log_receiver.port,
                        world_size=world_size,
                        rank=i + 1,
                        env_vars=(self.default_env_vars + self.extra_env_vars),
                        env_file=self.env_file,
                    ),
                    hostname=hostname,
                    ssh_config_file=self.ssh_config_file,
                )

            # initialize launcher-agent process group
            # ranks = (launcher, agent_{hostnames[0]}, ..., agent[-1])

            launcher_agent_group = LauncherAgentGroup(
                launcher_hostname=launcher_hostname,
                launcher_port=launcher_port,
                world_size=world_size,
                rank=0,
            )

            # build and sync payloads between launcher and agents

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

            # loop to monitor agent statuses (until failed or done)

            while True:
                # raises RuntimeError if communication timeout due to death of any agent
                agent_statuses = launcher_agent_group.sync_agent_statuses(status=None)

                # raises specific exception if any agent fails
                for s in agent_statuses:
                    for value in s.return_values:
                        if isinstance(value, WorkerException):
                            raise value.exception

                if all(s.state == "done" for s in agent_statuses):
                    break
        finally:
            if log_receiver is not None:
                log_receiver.shutdown()
                if log_process is not None:
                    log_receiver.server_close()
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

        return LaunchResult(hostnames=hostnames, agent_statuses=agent_statuses)


def launch(
    func: Callable,
    func_args: tuple[Any] | None = None,
    func_kwargs: dict[str, Any] | None = None,
    hostnames: list[str] | Literal["auto", "slurm"] = "auto",
    workers_per_host: int | list[int] | Literal["auto", "slurm"] = "auto",
    ssh_config_file: str | os.PathLike | None = None,
    backend: Literal["nccl", "gloo", "mpi", "ucc", "auto"] | None = "auto",
    timeout: int = 600,
    log_handlers: list[Handler] | Literal["auto"] | None = "auto",
    default_env_vars: tuple[str] = (  # pyright: ignore [reportAssignmentType]
        "PATH",
        "LD_LIBRARY",
        "LIBRARY_PATH",
        "PYTHON*",
        "CUDA*",
        "TORCH*",
        "PYTORCH*",
        "NCCL*",
    ),
    extra_env_vars: tuple[str] = (),
    env_file: str | os.PathLike | None = None,
) -> LaunchResult:
    """
    Launch a distributed PyTorch function on the specified nodes.

    :param func:
    :param func_args:
    :param func_kwargs:
    :param hostnames: Nodes to launch the function on. Default infers from a SLURM environment or runs on localhost.
    :param workers_per_host: Number of processes to run per node. Can define per node with :type:`list[int]`.
    :param ssh_config_file: An SSH configuration file for connecting to nodes, by default loads ``~/.ssh/config`` or ``/etc/ssh/ssh_config``.
    :param backend: `Backend <https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend>`_ to initialize worker process group with. Default uses NCCL (if GPUs available) or GLOO. Disabled by ``None``.
    :param timeout: Worker process group timeout (seconds).
    :param log_handlers: A list of handlers to manage agent and worker logs. Default uses an automatic basic logging scheme.
    :param default_env_vars: A list of environmental variables to be copied from the launcher process to workers. Allows for bash pattern matching syntax.
    :param extra_env_vars: Additional, user-specified variables to copy.
    :param env_file: A file (like ``.env``) with additional environment variables to copy.
    :raises RuntimeError: May fail if ``torch.distributed`` not available or communication timeout between nodes
    :raises Exception: Propagates exceptions raised in worker processes
    """  # noqa: E501
    return Launcher(
        hostnames=hostnames,
        workers_per_host=workers_per_host,
        ssh_config_file=ssh_config_file,
        backend=backend,
        timeout=timeout,
        log_handlers=log_handlers,
        default_env_vars=default_env_vars,
        extra_env_vars=extra_env_vars,
        env_file=env_file,
    ).run(func=func, func_args=func_args, func_kwargs=func_kwargs)


class LaunchResult:
    def __init__(self, hostnames: list[str], agent_statuses: list[AgentStatus]) -> None:
        self.hostnames: list[str] = hostnames
        self.return_values: list[list[Any]] = [s.return_values for s in agent_statuses]

    @overload
    def all(self) -> dict[str, list[Any]]:
        pass

    @overload
    def all(self, by: Literal["hostname"]) -> dict[str, list[Any]]:
        pass

    @overload
    def all(self, by: Literal["rank"]) -> list[Any]:
        pass

    def all(self, by: Literal["hostname", "rank"] = "hostname") -> dict[str, list[Any]] | list[Any]:
        """
        Get all worker return values by rank or hostname.

        :param by: Whether to aggregate all return values by hostname, or just output all of them in order of rank, defaults to ``'hostname'``
        """
        if by == "hostname":
            return dict(zip(self.hostnames, self.return_values))
        elif by == "rank":  # noqa: RET505
            return reduce(add, self.return_values)

        msg = "Invalid argument: expected by=('hostname' | 'rank')"
        raise TypeError(msg)

    def values(self, hostname: str) -> list[Any]:
        """
        Get worker return values for host ``hostname``.

        :param hostname: The host to get return values from
        """
        host_idx = self.hostnames.index(hostname)
        return self.return_values[host_idx]

    def value(self, rank: int) -> Any:
        """
        Get worker return value from global rank ``rank``.

        :param rank: Global worker rank to get return value from
        """
        if rank < 0:
            msg = f"Rank {rank} must be larger than 0"
            raise ValueError(msg)

        for values in self.return_values:
            if rank >= len(values):
                rank -= len(values)
            else:
                return values[rank]

        msg = f"Rank {rank} larger than world_size"
        raise ValueError(msg)

from __future__ import annotations

import fnmatch
import ipaddress
import itertools
import logging
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from logging import Handler
from multiprocessing import Process
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import fabric
import torch.distributed as dist

from .environment import auto_hosts, auto_workers, slurm_hosts, slurm_workers
from .logging_utils import LogRecordSocketReceiver, default_handlers
from .utils import (
    LauncherAgentGroup,
    LauncherPayload,
    WorkerException,
    get_open_port,
)


def resolve_hostnames(hostnames: list[str] | Literal["auto", "slurm"]) -> list[str]:
    if hostnames == "auto":
        return auto_hosts()
    elif hostnames == "slurm":
        return slurm_hosts()
    return hostnames


def resolve_workers_per_host(
    workers_per_host: int | list[int] | Literal["auto", "slurm"], num_hosts: int
) -> list[int]:
    if workers_per_host == "auto":
        workers_per_host = auto_workers()
    elif workers_per_host == "slurm":
        workers_per_host = slurm_workers()

    if isinstance(workers_per_host, int):
        workers_per_host = [workers_per_host] * num_hosts
    else:
        assert len(workers_per_host) == num_hosts

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

    log_receiver = LogRecordSocketReceiver(
        host=launcher_hostname,
        port=get_open_port(),
        handlers=log_handlers,
    )

    return log_receiver


def is_localhost(hostname_or_ip: str) -> bool:
    # check if host is "loopback" address (i.e. designated to send to self)
    try:
        ip = ipaddress.ip_address(hostname_or_ip)
    except ValueError:
        ip = ipaddress.ip_address(socket.gethostbyname(hostname_or_ip))
    if ip.is_loopback:
        return True
    # else compare local interface addresses between host and localhost
    host_addrs = [addr[4][0] for addr in socket.getaddrinfo(str(ip), None)]
    localhost_addrs = [addr[4][0] for addr in socket.getaddrinfo(socket.gethostname(), None)]
    return len(set(host_addrs) & set(localhost_addrs)) > 0


def execute_command(
    command: str,
    hostname: str,
    ssh_config_file: str | os.PathLike | None = None,
) -> None:
    if is_localhost(hostname):
        subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        with fabric.Connection(
            host=hostname, config=fabric.Config(runtime_ssh_path=ssh_config_file)
        ) as conn:
            conn.run(f"{command} >> /dev/null 2>&1 &", asynchronous=True)


def build_command(
    launcher_hostname: str,
    launcher_port: int,
    logger_port: int,
    world_size: int,
    rank: int,
    env_vars: Sequence[str],
    env_file: str | os.PathLike | None,
) -> str:
    current_dir = os.getcwd()

    env_exports = []
    for k, v in os.environ.items():
        if any(fnmatch.fnmatch(k, e) for e in env_vars):
            env_exports.append(f"{k}={v}")

    env_export_string = ""
    if len(env_exports) > 0:
        env_export_string = f"export {' '.join(env_exports)} && "

    env_file_string = ""
    if env_file is not None:
        env_file_string = f"source {env_file} && "

    return (
        f"cd {current_dir} && "
        f"{env_export_string}"
        f"{env_file_string}"
        f"{sys.executable} -u -m torchrunx "
        f"--launcher-hostname {launcher_hostname} "
        f"--launcher-port {launcher_port} "
        f"--logger-port {logger_port} "
        f"--world-size {world_size} "
        f"--rank {rank}"
    )


@dataclass
class Launcher:
    hostnames: list[str] | Literal["auto", "slurm"] = "auto"
    workers_per_host: int | list[int] | Literal["auto", "slurm"] = "auto"
    ssh_config_file: str | os.PathLike | None = None
    backend: Literal["mpi", "gloo", "nccl", "ucc", None] = None
    log_handlers: list[Handler] | Literal["auto"] | None = "auto"
    env_vars: Sequence[str] = (
        "PATH",
        "LD_LIBRARY",
        "LIBRARY_PATH",
        "PYTHON*",
        "CUDA*",
        "TORCH*",
        "PYTORCH*",
        "NCCL*",
    )
    env_file: str | os.PathLike | None = None
    timeout: int = 600

    def run(
        self,
        func: Callable,
        func_args: tuple[Any] | None = None,
        func_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, dict[int, Any]]:
        """
        Launch a distributed PyTorch function on the specified nodes. See :mod:`torchrunx.launch`

        :param func: The distributed function to call on all workers
        :type func: Callable
        :param func_args: Any positional arguments to be provided when calling ``func``
        :type func_args: tuple[Any]
        :param func_kwargs: Any keyword arguments to be provided when calling ``func``
        :type func_kwargs: dict[str, Any]
        :raises RuntimeError: May fail due to misconfiguration, or errors thrown by ``func``
        :return: A dictionary mapping worker ranks to their output
        :rtype: dict[int, Any]
        """
        if not dist.is_available():
            raise RuntimeError("The torch.distributed package is not available.")

        hostnames = resolve_hostnames(self.hostnames)
        workers_per_host = resolve_workers_per_host(self.workers_per_host, len(hostnames))

        launcher_hostname = socket.getfqdn()
        launcher_port = get_open_port()
        world_size = len(hostnames) + 1

        # start logging server

        log_receiver = build_logging_server(
            log_handlers=self.log_handlers,
            launcher_hostname=launcher_hostname,
            hostnames=hostnames,
            workers_per_host=workers_per_host,
            log_dir=Path(os.environ.get("TORCHRUNX_LOG_DIR", "torchrunx_logs")),
            log_level=logging._nameToLevel[os.environ.get("TORCHRUNX_LOG_LEVEL", "INFO")],
        )

        log_process = Process(
            target=log_receiver.serve_forever,
            daemon=True,
        )

        log_process.start()

        # start agents on each node

        for i, hostname in enumerate(hostnames):
            execute_command(
                command=build_command(
                    launcher_hostname=launcher_hostname,
                    launcher_port=launcher_port,
                    logger_port=log_receiver.port,
                    world_size=world_size,
                    rank=i + 1,
                    env_vars=self.env_vars,
                    env_file=self.env_file,
                ),
                hostname=hostname,
                ssh_config_file=self.ssh_config_file,
            )

        # initialize launcher–agent process group
        # ranks = (launcher, agent_{hostnames[0]}, ..., agent[-1])

        launcher_agent_group = LauncherAgentGroup(
            launcher_hostname=launcher_hostname,
            launcher_port=launcher_port,
            world_size=world_size,
            rank=0,
        )

        # build and sync payloads between launcher and agents

        _cumulative_workers = [0] + list(itertools.accumulate(workers_per_host))

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

        try:
            while True:
                agent_statuses = launcher_agent_group.sync_agent_statuses(status=None)
                # raises exception if communication timeout due to death of any agent

                for s in agent_statuses:
                    if s.state == "failed":
                        for value in s.return_values.values():
                            if isinstance(value, WorkerException):
                                raise value.exception

                if all(s.state == "done" for s in agent_statuses):
                    break

        except:
            # cleanup: SIGTERM all agents
            for agent_payload, agent_hostname in zip(agent_payloads, hostnames):
                execute_command(
                    command=f"kill {agent_payload.process_id}",
                    hostname=agent_hostname,
                    ssh_config_file=self.ssh_config_file,
                )
            raise
        finally:
            log_receiver.shutdown()
            log_receiver.server_close()
            log_process.kill()
            dist.destroy_process_group()

        return {
            hostname: agent_status.return_values
            for hostname, agent_status in zip(hostnames, agent_statuses)
        }


def launch(
    func: Callable,
    func_args: tuple[Any] | None = None,
    func_kwargs: dict[str, Any] | None = None,
    hostnames: list[str] | Literal["auto", "slurm"] = "auto",
    workers_per_host: int | list[int] | Literal["auto", "slurm"] = "auto",
    ssh_config_file: str | os.PathLike | None = None,
    backend: Literal["mpi", "gloo", "nccl", "ucc", None] = None,
    log_handlers: list[Handler] | Literal["auto"] = "auto",
    env_vars: Sequence[str] = (
        "PATH",
        "LD_LIBRARY",
        "LIBRARY_PATH",
        "PYTHON*",
        "CUDA*",
        "TORCH*",
        "PYTORCH*",
        "NCCL*",
    ),
    env_file: str | os.PathLike | None = None,
    timeout: int = 600,
) -> dict[str, dict[int, Any]]:
    """
    Launch a distributed PyTorch function on the specified nodes.

    :param func: The distributed function to call on all workers
    :type func: Callable
    :param func_args: Any positional arguments to be provided when calling ``func``
    :type func_args: tuple[Any]
    :param func_kwargs: Any keyword arguments to be provided when calling ``func``
    :type func_kwargs: dict[str, Any]
    :param auto: Automatically determine allocation sizes, supports Slurm allocation. ``hostnames`` and ``workers_per_host`` are automatically assigned if they're set to ``None``, defaults to None
    :type auto: bool, optional
    :param hostnames: A list of node hostnames to start workers on, defaults to ["localhost"]
    :type hostnames: list[str] | Literal["auto", "slurm"] | None, optional
    :param workers_per_host: The number of workers per node. Providing an ``int`` implies all nodes should have ``workers_per_host`` workers, meanwhile providing a list causes node ``i`` to have ``worker_per_host[i]`` workers, defaults to 1
    :type workers_per_host: int | list[int] | Literal["auto", "slurm"] | None, optional
    :param ssh_config_file: An SSH configuration file to use when connecting to nodes, defaults to None
    :type ssh_config_file: str | os.PathLike | None, optional
    :param backend: A ``torch.distributed`` `backend string <https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend>`_, defaults to None
    :type backend: Literal['mpi', 'gloo', 'nccl', 'ucc', None], optional
    :param log_handlers: A list of handlers to manage agent and worker logs, defaults to []
    :type log_handlers: list[Handler] | Literal["auto"], optional
    :param env_vars: A list of environmental variables to be copied from the launcher environment to workers. Allows for bash pattern matching syntax, defaults to ["PATH", "LD_LIBRARY", "LIBRARY_PATH", "PYTHON*", "CUDA*", "TORCH*", "PYTORCH*", "NCCL*"]
    :type env_vars: list[str], optional
    :param env_file: An additional environment file that will be sourced prior to executing ``func``, defaults to None
    :type env_file: str | os.PathLike | None, optional
    :param timeout: Worker process group timeout, defaults to 600
    :type timeout: int, optional
    :raises RuntimeError: May fail due to misconfiguration, or errors thrown by ``func``
    :return: A dictionary mapping worker ranks to their output
    :rtype: dict[int, Any]
    """  # noqa: E501
    return Launcher(
        hostnames=hostnames,
        workers_per_host=workers_per_host,
        ssh_config_file=ssh_config_file,
        backend=backend,
        log_handlers=log_handlers,
        env_vars=env_vars,
        env_file=env_file,
        timeout=timeout,
    ).run(func=func, func_args=func_args, func_kwargs=func_kwargs)

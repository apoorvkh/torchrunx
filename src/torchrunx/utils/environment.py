"""Utilities for determining hosts and workers in environment."""

from __future__ import annotations

from typing import Literal, Union

from typing_extensions import TypeAlias

__all__ = [
    "auto_hosts",
    "build_launch_command",
    "execute_command",
    "get_gpus_per_host",
    "in_slurm_job",
    "slurm_hosts",
]

import ipaddress
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import fabric

Hostnames: TypeAlias = list[str]
WorkersPerHost: TypeAlias = list[int]
Backend: TypeAlias = Union[Literal["nccl", "gloo", "mpi", "ucc"], None]


def resolve_environment(
    hostnames: list[str] | Literal["auto", "slurm"],
    workers_per_host: int | list[int] | Literal["auto"],
    backend: Literal["nccl", "gloo", "mpi", "ucc", "auto"] | None,
    ssh_config_file: str | os.PathLike | None,
) -> tuple[Hostnames, WorkersPerHost, Backend]:
    if hostnames == "auto":
        hostnames = auto_hosts()
    elif hostnames == "slurm":
        hostnames = slurm_hosts()

    if isinstance(workers_per_host, int):
        workers_per_host = [workers_per_host] * len(hostnames)

    if workers_per_host == "auto" or backend == "auto":
        gpus_per_host: list[int] = get_gpus_per_host(hostnames, ssh_config_file)
        gpus_on_every_host: bool = all(g > 0 for g in gpus_per_host)

        if workers_per_host == "auto":
            if not gpus_on_every_host:
                msg = 'workers_per_host="auto", but no GPUs detected on at least one host.'
                raise RuntimeError(msg)
            workers_per_host = gpus_per_host

        if backend == "auto":
            backend = "nccl" if gpus_per_host else "gloo"

    return hostnames, workers_per_host, backend


def auto_hosts() -> list[str]:
    """Automatically determine hostnames to launch to."""
    if in_slurm_job():
        return slurm_hosts()
    return ["localhost"]


def in_slurm_job() -> bool:
    """Check if current process is running in a Slurm allocation."""
    return "SLURM_JOB_ID" in os.environ or "SLURM_JOBID" in os.environ


def slurm_hosts() -> list[str]:
    """Retrieves hostnames of Slurm-allocated nodes."""
    if not in_slurm_job():
        msg = "Not in a SLURM job"
        raise RuntimeError(msg)

    return subprocess.check_output(["scontrol", "show", "hostnames"]).decode().strip().split("\n")


def get_gpus_per_host(hostnames: list[str], ssh_config_file: str | os.PathLike | None) -> list[int]:
    """Count the number of GPUs on each host."""
    python = shlex.quote(sys.executable)
    command = f"{python} -c \"import torch; print(torch.cuda.device_count(), end='')\""
    return [
        int(
            execute_command(
                command, hostname, ssh_config_file=ssh_config_file, return_stdout_stderr=True
            )[0]
        )
        for hostname in hostnames
    ]


def build_launch_command(
    launcher_hostname: str,
    launcher_port: int,
    logger_port: int,
    world_size: int,
    rank: int,
    env_vars: dict[str, str],
    env_file: str | os.PathLike | None,
) -> str:
    """Generator for command to launch torchrunx on an agent."""
    # shlex.quote prevents shell injection here (resolves S602 in execute_command)

    commands = []

    commands.append(f"cd {shlex.quote(str(Path.cwd()))}")

    env_exports = [shlex.quote(f"{k}={v}") for k, v in env_vars.items()]
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
    *,
    ssh_config_file: str | os.PathLike | None = None,
    return_stdout_stderr: bool = False,
) -> tuple[str, str]:
    """Run a command on local or remote host (using SSH)."""
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
        process = subprocess.Popen(  # noqa: S602
            command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if return_stdout_stderr:
            stdout, stderr = process.communicate()
            return stdout, stderr
    else:
        runtime_ssh_path = ssh_config_file
        if isinstance(ssh_config_file, os.PathLike):
            runtime_ssh_path = str(ssh_config_file)

        with fabric.Connection(
            host=hostname,
            config=fabric.Config(runtime_ssh_path=runtime_ssh_path),
        ) as conn:
            promise = conn.run(command, asynchronous=True, hide=True)

            if return_stdout_stderr:
                results = promise.join()
                return results.stdout, results.stderr

    return ("", "")

"""Utilities for building a Launcher from argparse command-line arguments."""

from __future__ import annotations

__all__ = ["add_torchrunx_argument_group", "launcher_from_args"]

from argparse import ArgumentParser, Namespace
from typing import Literal

from torchrunx import DEFAULT_ENV_VARS_FOR_COPY, Launcher


def add_torchrunx_argument_group(parser: ArgumentParser) -> None:
    """Add an argument group for torchrunx.Launcher to an ArgumentParser."""
    group = parser.add_argument_group("torchrunx")

    group.add_argument(
        "--hostnames",
        type=str,
        nargs="+",
        default="auto",
        help="Nodes to launch the function on. Default: 'auto'. Use 'slurm' to infer from SLURM.",
    )

    group.add_argument(
        "--workers-per-host",
        type=str,
        nargs="+",
        default="gpu",
        help="Processes to run per node. Can be 'cpu', 'gpu', or list[int]. Default: 'gpu'.",
    )

    group.add_argument(
        "--ssh-config-file",
        type=str,
        default=None,
        help="Path to SSH config file. Default: '~/.ssh/config' or '/etc/ssh/ssh_config'.",
    )

    group.add_argument(
        "--backend",
        type=str,
        choices=["nccl", "gloo", "mpi", "ucc", "None"],
        default="nccl",
        help="For worker process group. Default: 'nccl'. Use 'gloo' for CPU. 'None' to disable.",
    )

    group.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Worker process group timeout in seconds. Default: 600.",
    )

    group.add_argument(
        "--agent-timeout",
        type=int,
        default=30,
        help="Agent communication timeout in seconds. Default: 30.",
    )

    group.add_argument(
        "--copy-env-vars",
        type=str,
        nargs="+",
        default=DEFAULT_ENV_VARS_FOR_COPY,
        help="Environment variables to copy to workers. Supports Unix pattern matching.",
    )

    group.add_argument(
        "--extra-env-vars",
        type=str,
        nargs="*",
        default=None,
        help="Additional environment variables as key=value pairs.",
    )

    group.add_argument(
        "--env-file", type=str, default=None, help="Path to a .env file with environment variables."
    )


def launcher_from_args(args: Namespace) -> Launcher:
    """Create a torchrunx.Launcher from argparse.Namespace."""
    _hostnames: list[str] = args.hostnames
    hostnames: list[str] | Literal["auto", "slurm"]
    if _hostnames == ["auto"]:
        hostnames = "auto"
    elif _hostnames == ["slurm"]:
        hostnames = "slurm"
    else:
        hostnames = _hostnames

    _workers_per_host: list[str] = args.workers_per_host
    workers_per_host: int | list[int] | Literal["cpu", "gpu"]

    if _workers_per_host == ["cpu"]:
        workers_per_host = "cpu"
    elif _workers_per_host == ["gpu"]:
        workers_per_host = "gpu"
    elif len(_workers_per_host) == 1:
        workers_per_host = int(_workers_per_host[0])
    else:
        workers_per_host = [int(w) for w in _workers_per_host]

    ssh_config_file: str | None = args.ssh_config_file

    _backend: str = args.backend
    backend: Literal["nccl", "gloo", "mpi", "ucc"] | None
    if _backend == "None":  # noqa: SIM108
        backend = None
    else:
        backend = _backend  # pyright: ignore [reportAssignmentType]

    timeout: int = args.timeout
    agent_timeout: int = args.agent_timeout

    copy_env_vars: tuple[str, ...] = tuple(args.copy_env_vars)

    _extra_env_vars: list[str] | None = args.extra_env_vars
    extra_env_vars: dict[str, str] | None
    if _extra_env_vars is not None:
        extra_env_vars = dict(var.split("=", 1) for var in _extra_env_vars)
    else:
        extra_env_vars = None

    env_file: str | None = args.env_file

    return Launcher(
        hostnames=hostnames,
        workers_per_host=workers_per_host,
        ssh_config_file=ssh_config_file,
        backend=backend,
        timeout=timeout,
        agent_timeout=agent_timeout,
        copy_env_vars=copy_env_vars,
        extra_env_vars=extra_env_vars,
        env_file=env_file,
    )

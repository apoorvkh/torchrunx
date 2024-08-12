from __future__ import annotations

import os
import subprocess

import torch


def in_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ


def slurm_hosts() -> list[str]:
    """Retrieves hostnames of Slurm-allocated nodes.

    :return: Hostnames of nodes in current Slurm allocation
    :rtype: list[str]
    """
    # TODO: sanity check SLURM variables, commands
    assert in_slurm_job()
    return (
        subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        .decode()
        .strip()
        .split("\n")
    )


def slurm_workers() -> int:
    """
    |  Determines number of workers per node in current Slurm allocation using
    |  the ``SLURM_JOB_GPUS`` or ``SLURM_CPUS_ON_NODE`` environmental variables.

    :return: The implied number of workers per node
    :rtype: int
    """
    # TODO: sanity check SLURM variables, commands
    assert in_slurm_job()
    if "SLURM_JOB_GPUS" in os.environ:
        # TODO: is it possible to allocate uneven GPUs across nodes?
        return len(os.environ["SLURM_JOB_GPUS"].split(","))
    else:
        # TODO: should we assume that we plan to do one worker per CPU?
        return int(os.environ["SLURM_CPUS_ON_NODE"])


def auto_hosts() -> list[str]:
    """
    Automatically determine hostname list

    :return: Hostnames in Slurm allocation, or ['localhost']
    :rtype: list[str]
    """
    if in_slurm_job():
        slurm_hosts()

    return ["localhost"]


def auto_workers() -> int:
    """
    Automatically determine number of workers per host

    :return: Workers per host
    :rtype: int
    """
    if in_slurm_job():
        return slurm_workers()

    return torch.cuda.device_count() or os.cpu_count() or 1

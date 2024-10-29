"""Utilities for determining hosts and workers in environment."""

from __future__ import annotations

__all__ = ["in_slurm_job", "slurm_hosts", "slurm_workers", "auto_hosts", "auto_workers"]

import os
import subprocess

import torch


def in_slurm_job() -> bool:
    """Check if current process is running in a Slurm allocation."""
    return "SLURM_JOB_ID" in os.environ


def slurm_hosts() -> list[str]:
    """Retrieves hostnames of Slurm-allocated nodes."""
    # TODO: sanity check SLURM variables, commands
    if not in_slurm_job():
        msg = "Not in a SLURM job"
        raise RuntimeError(msg)
    return (
        subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        .decode()
        .strip()
        .split("\n")
    )


def slurm_workers() -> int:
    """Determines number of workers per node in current Slurm allocation."""
    # TODO: sanity check SLURM variables, commands
    if not in_slurm_job():
        msg = "Not in a SLURM job"
        raise RuntimeError(msg)

    if "SLURM_JOB_GPUS" in os.environ:
        # TODO: is it possible to allocate uneven GPUs across nodes?
        return len(os.environ["SLURM_JOB_GPUS"].split(","))
    if "SLURM_GPUS_PER_NODE" in os.environ:
        return int(os.environ["SLURM_GPUS_PER_NODE"])

    return int(os.environ["SLURM_CPUS_ON_NODE"])


def auto_hosts() -> list[str]:
    """Automatically determine hostnames to launch to."""
    if in_slurm_job():
        return slurm_hosts()

    return ["localhost"]


def auto_workers() -> int:
    """Automatically determine workers per host from SLURM or based on GPU/CPU count."""
    if in_slurm_job():
        return slurm_workers()

    return torch.cuda.device_count() or os.cpu_count() or 1

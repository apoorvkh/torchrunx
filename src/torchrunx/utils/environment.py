"""Utilities for determining hosts and workers in environment."""

from __future__ import annotations

__all__ = ["auto_hosts", "in_slurm_job", "slurm_hosts"]

import os
import subprocess


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

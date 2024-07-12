from __future__ import annotations

from .launcher import launch


def slurm_hosts() -> list[str]:
    """Retrieves hostnames of Slurm-allocated nodes.

    :return: Hostnames of nodes in current Slurm allocation
    :rtype: list[str]
    """
    import os
    import subprocess

    # TODO: sanity check SLURM variables, commands
    assert "SLURM_JOB_ID" in os.environ
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
    import os

    # TODO: sanity check SLURM variables, commands
    assert "SLURM_JOB_ID" in os.environ
    if "SLURM_JOB_GPUS" in os.environ:
        # TODO: is it possible to allocate uneven GPUs across nodes?
        return len(os.environ["SLURM_JOB_GPUS"].split(","))
    else:
        # TODO: should we assume that we plan to do one worker per CPU?
        return int(os.environ["SLURM_CPUS_ON_NODE"])


__all__ = ["launch", "slurm_hosts", "slurm_workers"]

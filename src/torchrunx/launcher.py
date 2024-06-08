from __future__ import annotations

import itertools
import os
import socket
import subprocess
import sys
from functools import partial
from typing import Any, Callable, Literal

import torch.distributed as dist

from .utils import (
    LaunchConfig,
    LauncherAgentGroup,
    execute_command,
    get_open_port,
)


def launch(
    func: Callable,
    func_kwargs: dict[str, Any],
    hostnames: list[str] = ["localhost"],
    workers_per_host: int | list[int] | None = 1,
    use_slurm: bool = False,
    visible_devices_per_host: list[list[int]] | None = None,  # TODO
    ssh_config_file: str | os.PathLike | None = None,
    backend: Literal["mpi", "gloo", "nccl", "ucc"] | None = None,
    log_dir: str = "parallel_processing.log",  # TODO: use
):
    if not dist.is_available():
        raise RuntimeError("The torch.distributed package is not available.")

    # parse arguments

    if use_slurm:
        assert "SLURM_JOB_ID" in os.environ
        hostnames = (
            subprocess.check_output(
                ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
            )
            .decode()
            .strip()
            .split("\n")
        )
        if "SLURM_JOB_GPUS" in os.environ:
            # is it possible to allocate uneven GPUs across nodes?
            workers_per_host = len(os.environ["SLURM_JOB_GPUS"].split(","))
        else:
            # should we assume that we plan to do one worker per CPU?
            workers_per_host = int(os.environ["SLURM_CPUS_ON_NODE"])

    num_hosts = len(hostnames)

    if workers_per_host is not None and isinstance(workers_per_host, int):
        workers_per_host = [workers_per_host] * num_hosts

    if visible_devices_per_host is not None:
        # TODO: slurm case
        workers_per_host = [len(indices) for indices in visible_devices_per_host]

    assert workers_per_host is not None
    assert len(workers_per_host) == num_hosts

    world_size = num_hosts + 1

    launcher_hostname = socket.gethostname()
    launcher_ip = socket.gethostbyname(launcher_hostname)
    launcher_port = get_open_port()

    # start agents on each node
    for i, hostname in enumerate(hostnames):
        execute_command(
            command=(
                f"{sys.executable} -u -m torchrunx "
                f"--world-size {world_size} "
                f"--rank {i+1} "
                f"--launcher-ip {launcher_ip} "
                f"--launcher-port {launcher_port}"
            ),
            hostname=hostname,
            ssh_config_file=ssh_config_file,
        )

    # initialize launcher–agent process group
    # ranks = (launcher, agent_0, ..., agent_{num_hosts-1})

    launcher_group = LauncherAgentGroup(
        world_size=world_size,
        rank=0,
        launcher_hostname=launcher_hostname,
        launcher_port=launcher_port,
    )

    # build LaunchConfig
    cumulative_workers = [0] + list(itertools.accumulate(workers_per_host))
    worker_global_ranks = [
        list(range(cumulative_workers[n], cumulative_workers[n + 1])) for n in range(num_hosts)
    ]

    config = LaunchConfig(
        fn=partial(func, **func_kwargs),
        world_size=cumulative_workers[-1],
        node_worker_ranks=worker_global_ranks,
        backend=backend,
    )

    # communicate parameters with agents
    launcher_group.send_launch_config(config)
    _ = launcher_group.sync_main_agent_ip_port()
    agent_pids = launcher_group.recv_agent_process_ids()

    # start monitoring loop
    while True:
        try:
            agent_statuses = launcher_group.all_gather_agent_statuses(status=None)
        except:
            # kill all agents (most should be dead but some could be hanging)
            for pid, ip_forgn in zip(agent_pids, hostnames):
                execute_command(
                    command=f"kill {pid}",
                    hostname=ip_forgn,
                    ssh_config_file=ssh_config_file,
                )
            # TODO: can we extract more info for this error?
            raise RuntimeError("One or more agents encountered an error.")

        if any([s.is_failed() for s in agent_statuses]):
            # terminate - the agents should also be exiting
            e = ""
            for i, s in enumerate(agent_statuses):
                if s.is_failed():
                    for k, v in s.failures.items():
                        e += f"Node {i}, local worker {k} exited with error: {v.message['message']}\n"
                        e += f"{v.message['extraInfo']['py_callstack']}\n\n"
            raise RuntimeError(e)

        # else, check if everything's done
        if all(map(lambda s: s.is_done(), agent_statuses)):
            # we can exit loop and gather return values
            break

    # print stdouts and stderrs
    r = 0
    for node, status in enumerate(agent_statuses):
        for worker in status.stdouts:
            if status.stdouts[worker] != "":
                print(
                    f"Node {node}, worker {worker} (rank {r}) stdout:\n{status.stdouts[worker]}",
                    file=sys.stdout,
                )
            if status.stderrs[worker] != "":
                print(
                    f"Node {node}, worker {worker} (rank {r}) stderr:\n{status.stderrs[worker]}",
                    file=sys.stderr,
                )
            r += 1

    # wait for return values
    try:
        outputs = launcher_group.recv_return_values()
    except:
        for pid, ip_forgn in zip(agent_pids, hostnames):
            execute_command(
                command=f"kill {pid}",
                hostname=ip_forgn,
                ssh_config_file=ssh_config_file,
            )
        # TODO: can we extract more info for this error?
        raise RuntimeError("One or more agents encountered an error.")

    # gather return values in {worker_rank: worker_return_value} format, and return
    result = {}
    for d in outputs:
        result.update(d)
    return result

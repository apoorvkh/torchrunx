from __future__ import annotations

import datetime
import fnmatch
import itertools
import os
import socket
import subprocess
import sys
from collections import ChainMap
from functools import partial
from multiprocessing import Process
from pathlib import Path
from typing import Any, Callable, Literal

import torch.distributed as dist

from .utils import (
    AgentPayload,
    AgentStatus,
    LauncherAgentGroup,
    LauncherPayload,
    execute_command,
    get_open_port,
    monitor_log,
)


def launch(
    func: Callable,
    func_kwargs: dict[str, Any],
    hostnames: list[str] = ["localhost"],
    workers_per_host: int | list[int] = 1,
    use_slurm: bool = False,
    ssh_config_file: str | os.PathLike | None = None,
    backend: Literal["mpi", "gloo", "nccl", "ucc", None] = None,
    log_dir: os.PathLike | str = "./logs",
    env_vars: list[str] = [
        "PATH",
        "LD_LIBRARY",
        "LIBRARY_PATH",
        "PYTHON*",
        "CUDA*",
        "TORCH*",
        "PYTORCH*",
        "NCCL*",
    ],
    env_file: str | os.PathLike | None = None,
):
    if not dist.is_available():
        raise RuntimeError("The torch.distributed package is not available.")

    if use_slurm:
        # TODO: sanity check these variables, commands
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
            # TODO: is it possible to allocate uneven GPUs across nodes?
            workers_per_host = len(os.environ["SLURM_JOB_GPUS"].split(","))
        else:
            # TODO: should we assume that we plan to do one worker per CPU?
            workers_per_host = int(os.environ["SLURM_CPUS_ON_NODE"])

    num_hosts = len(hostnames)

    if isinstance(workers_per_host, int):
        workers_per_host = [workers_per_host] * num_hosts

    assert workers_per_host is not None
    assert len(workers_per_host) == num_hosts

    # launch command

    env_export_string = " ".join(
        f'{k}="{v}"' for k, v in os.environ.items() if any(fnmatch.fnmatch(k, e) for e in env_vars)
    )
    if env_export_string != "":
        env_export_string = f"export {env_export_string} && "

    env_file_string = f"source {env_file} && " if env_file is not None else ""

    launcher_hostname = socket.getfqdn()
    launcher_port = get_open_port()
    world_size = num_hosts + 1  # launcher + agents

    command = (
        f"cd {os.getcwd()} && "
        f"{env_export_string}"
        f"{env_file_string}"
        f"{sys.executable} -u -m torchrunx "
        f"--launcher-hostname {launcher_hostname} "
        f"--launcher-port {launcher_port} "
        f"--world-size {world_size} "
        # rank set in the loop below
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H%M%S")

    # start process to read from agent 0 log
    print_process = Process(target=monitor_log, args=(log_dir / f"{timestamp}_{hostnames[0]}.log",))
    print_process.start()

    # start agents on each node
    for i, hostname in enumerate(hostnames):
        execute_command(
            command=f"{command} --rank {i+1}",
            hostname=hostname,
            ssh_config_file=ssh_config_file,
            outfile=os.fspath(log_dir / f"{timestamp}_{hostname}.log"),
        )

    # initialize launcher–agent process group
    # ranks = (launcher, agent_0, ..., agent_{num_hosts-1})

    launcher_agent_group = LauncherAgentGroup(
        launcher_hostname=launcher_hostname,
        launcher_port=launcher_port,
        world_size=world_size,
        rank=0,
    )

    # build launcher payload (to share with agents)

    cumulative_workers = [0] + list(itertools.accumulate(workers_per_host))
    worker_world_size = cumulative_workers[-1]
    worker_global_ranks = [
        list(range(cumulative_workers[n], cumulative_workers[n + 1])) for n in range(num_hosts)
    ]  # list of worker ranks per host

    payload = LauncherPayload(
        fn=partial(func, **func_kwargs),
        worker_world_size=worker_world_size,
        worker_global_ranks=worker_global_ranks,
        backend=backend,
        log_dir=log_dir,
        log_prefix=timestamp,
        hostnames=hostnames,
    )

    # sync payloads; get PIDs of agents

    agent_payloads: list[AgentPayload] = launcher_agent_group.sync_payloads(payload=payload)[1:]  # pyright: ignore[reportAssignmentType]
    agent_pids = [p.process_id for p in agent_payloads]

    # loop to monitor agent statuses
    # kill all agent processes if timeout
    # print failures

    while True:
        try:
            agent_statuses = launcher_agent_group.sync_agent_statuses(status=AgentStatus())
        except Exception:  # TODO: should we wrap the "while True" with this?
            # on launcher_agent_group timeout: kill all agent processes
            for agent_pid, agent_hostname in zip(agent_pids, hostnames):
                execute_command(
                    command=f"kill {agent_pid}",
                    hostname=agent_hostname,
                    ssh_config_file=ssh_config_file,
                )
            raise

        if all(s.is_done() for s in agent_statuses):
            break

        if any(s.is_failed() for s in agent_statuses):
            # TODO: cleaner way to print these?
            e = ""
            for i, s in enumerate(agent_statuses):
                if s is not None and s.is_failed():
                    for k, v in s.failures.items():
                        e += f"Node {i}, local worker {k} exited with error: "
                        if isinstance(v.message, str):
                            e += f"{v.message}\n"
                        else:
                            e += f"{v.message['message']}\n"
                            e += f"{v.message['extraInfo']['py_callstack']}\n\n"
            raise RuntimeError(e)

    # terminate and return values

    print_process.terminate()

    return_values: dict[int, Any] = dict(ChainMap(*[s.return_values for s in agent_statuses]))
    return return_values

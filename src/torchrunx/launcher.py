from __future__ import annotations

import fnmatch
import itertools
import os
import socket
import subprocess
import sys
from collections import ChainMap
from contextlib import contextmanager
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
    random_log_dir,
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
    log_dir: str = "./logs",
    clone_env_vars: list[str] = ["PYTHON*", "CUDA*", "TORCH*", "PYTORCH*", "NCCL*"],
    env_file: str | os.PathLike | None = None,
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

    full_log_dir = random_log_dir(Path(os.path.abspath(log_dir)))
    full_log_dir.mkdir(parents=True)

    explicit_env_vars = ["PATH", "LD_LIBRARY", "LIBRARY_PATH"]
    env_export_string = " ".join(
        f'{k}="{v}"'
        for k, v in os.environ.items()
        if any(fnmatch.fnmatch(k, e) for e in clone_env_vars + explicit_env_vars)
    )
    if env_export_string != "":
        env_export_string = f"export {env_export_string} && "
    env_file_string = f"source {env_file} && " if env_file is not None else ""

    # start agents on each node
    for i, hostname in enumerate(hostnames):
        execute_command(
            command=(
                f"cd {os.getcwd()} && "
                f"{env_export_string}"
                f"{env_file_string}"
                f"{sys.executable} -u -m torchrunx "
                f"--world-size {world_size} "
                f"--rank {i+1} "
                f"--launcher-ip {launcher_ip} "
                f"--launcher-port {launcher_port} "
                f"--log-dir {os.fspath(full_log_dir)}"
            ),
            hostname=hostname,
            ssh_config_file=ssh_config_file,
            outfile=os.fspath(full_log_dir.joinpath(f"agent_{i}.log")),
        )

    # initialize launcher–agent process group
    # ranks = (launcher, agent_0, ..., agent_{num_hosts-1})

    launcher_group = LauncherAgentGroup(
        world_size=world_size,
        rank=0,
        launcher_hostname=launcher_hostname,
        launcher_port=launcher_port,
    )

    cumulative_workers = [0] + list(itertools.accumulate(workers_per_host))
    worker_global_ranks = [
        list(range(cumulative_workers[n], cumulative_workers[n + 1])) for n in range(num_hosts)
    ]

    payload = LauncherPayload(
        fn=partial(func, **func_kwargs),
        worker_world_size=cumulative_workers[-1],
        worker_global_ranks=worker_global_ranks,
        backend=backend,
    )

    agent_payloads: list[AgentPayload] = launcher_group.sync_payloads(payload=payload)[1:]  # pyright: ignore[reportAssignmentType]
    agent_pids = [p.process_id for p in agent_payloads]

    # start process to read from agent 0 log

    @contextmanager
    def print_agent_logs():
        print_process = Process(target=monitor_log, args=(full_log_dir / "agent_0.log",))
        print_process.start()
        try:
            yield
        finally:
            print_process.terminate()

    # start monitoring loop
    with print_agent_logs():
        while True:
            try:
                agent_statuses = launcher_group.sync_agent_statuses(status=AgentStatus())
            except Exception:
                # force kill all agents
                for agent_pid, agent_hostname in zip(agent_pids, hostnames):
                    execute_command(
                        command=f"kill {agent_pid}",
                        hostname=agent_hostname,
                        ssh_config_file=ssh_config_file,
                    )
                raise

            if any(s.is_failed() for s in agent_statuses):
                e = ""
                for i, s in enumerate(agent_statuses):
                    if s is not None and s.is_failed():
                        for k, v in s.failures.items():
                            e += f"Node {i}, local worker {k} exited with error: {v.message['message']}\n"  # type: ignore
                            e += f"{v.message['extraInfo']['py_callstack']}\n\n"  # type: ignore
                raise RuntimeError(e)
            elif all(s.is_done() for s in agent_statuses):
                break

    return_values: dict[int, Any] = dict(ChainMap(*[s.return_values for s in agent_statuses]))
    return return_values

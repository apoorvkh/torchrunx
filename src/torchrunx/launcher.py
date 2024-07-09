from __future__ import annotations

import datetime
import fnmatch
import itertools
import os
import socket
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
) -> dict[int, Any]:
    """
    Launch a distributed pytorch function on the specified nodes.

    :param func: The distributed function to call on all workers
    :type func: Callable
    :param func_kwargs: Any keyword arguments to be provided when calling ``func``
    :type func_kwargs: dict[str, Any]
    :param hostnames: A list of node hostnames to start workers on, defaults to ["localhost"]
    :type hostnames: list[str], optional
    :param workers_per_host: The number of workers per node. Providing an ``int`` implies all nodes should have ``workers_per_host`` workers, meanwhile providing a list causes node ``i`` to have ``worker_per_host[i]`` workers, defaults to 1
    :type workers_per_host: int | list[int], optional
    :param ssh_config_file: An SSH configuration file to use when connecting to nodes, defaults to None
    :type ssh_config_file: str | os.PathLike | None, optional
    :param backend: A ``torch.distributed`` `backend string <https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend>`_, defaults to None
    :type backend: Literal['mpi', 'gloo', 'nccl', 'ucc', None], optional
    :param log_dir: A directory in which logs should be written, defaults to "./logs"
    :type log_dir: os.PathLike | str, optional
    :param env_vars: A list of environmental variables to be copied from the launcher environment to workers. Allows for bash pattern matching syntax, defaults to ["PATH", "LD_LIBRARY", "LIBRARY_PATH", "PYTHON*", "CUDA*", "TORCH*", "PYTORCH*", "NCCL*"]
    :type env_vars: list[str], optional
    :param env_file: An additional environment file that will be sourced prior to executing ``func``, defaults to None
    :type env_file: str | os.PathLike | None, optional
    :raises RuntimeError: May fail due to misconfiguration, or errors thrown by ``func``
    :return: A dictionary mapping worker ranks to their output
    :rtype: dict[int, Any]
    """  # noqa: E501
    if not dist.is_available():
        raise RuntimeError("The torch.distributed package is not available.")

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
    agent_log_files = [log_dir / f"{timestamp}_{hostname}.log" for hostname in hostnames]

    # start process to read from agent 0 log
    print_process = Process(target=monitor_log, args=(agent_log_files[0],))
    print_process.start()

    # start agents on each node
    for i, hostname in enumerate(hostnames):
        execute_command(
            command=f"{command} --rank {i+1}",
            hostname=hostname,
            ssh_config_file=ssh_config_file,
            outfile=agent_log_files[i],
        )

    # initialize launcher–agent process group
    # ranks = (launcher, agent_0, ..., agent_{num_hosts-1})

    launcher_agent_group = LauncherAgentGroup(
        launcher_hostname=launcher_hostname,
        launcher_port=launcher_port,
        world_size=world_size,
        rank=0,
    )

    # build and sync payloads between launcher and agents

    cumulative_workers = [0] + list(itertools.accumulate(workers_per_host))
    worker_world_size = cumulative_workers[-1]
    worker_global_ranks = [  # list of worker ranks per host
        list(range(cumulative_workers[n], cumulative_workers[n + 1])) for n in range(num_hosts)
    ]
    worker_log_files = [
        [
            log_dir / f"{timestamp}_{hostname}_{local_rank}.log"
            for local_rank in range(workers_per_host[i])
        ]
        for i, hostname in enumerate(hostnames)
    ]

    payload = LauncherPayload(
        fn=partial(func, **func_kwargs),
        hostnames=hostnames,
        worker_world_size=worker_world_size,
        worker_global_ranks=worker_global_ranks,
        worker_log_files=worker_log_files,
        backend=backend,
    )

    agent_payloads: list[AgentPayload] = launcher_agent_group.sync_payloads(payload=payload)[1:]  # pyright: ignore[reportAssignmentType]
    agent_pids = [p.process_id for p in agent_payloads]

    # loop to monitor agent statuses (until failed or done)
    try:
        while True:
            agent_statuses = launcher_agent_group.sync_agent_statuses(status=AgentStatus())

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
    except:
        # kill all agents
        for agent_pid, agent_hostname in zip(agent_pids, hostnames):
            execute_command(
                command=f"kill {agent_pid}",
                hostname=agent_hostname,
                ssh_config_file=ssh_config_file,
            )
        raise
    #

    print_process.terminate()  # TODO: or close?
    return_values: dict[int, Any] = dict(ChainMap(*[s.return_values for s in agent_statuses]))
    return return_values

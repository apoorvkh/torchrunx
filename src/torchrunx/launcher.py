from __future__ import annotations
from typing import Literal, Any
import itertools

from dataclasses import dataclass
import os
import sys
import socket
from functools import partial
from typing import Callable
from enum import Enum
from datetime import timedelta

from torchrunx.utils import Serializable, get_open_port, execute_ssh_command, broadcast, gather, all_gather

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.api import RunProcsResult


@dataclass
class LaunchConfig(Serializable):
    fn: Callable
    world_size: int
    node_worker_ranks: list[list[int]]
    backend: Literal["mpi", "gloo", "nccl", "ucc"] | None


class Status(Enum):
    RUNNING = 1
    DONE = 2
    FAILED = 3


class AgentStatus:
    def __init__(self, result: RunProcsResult, dummy=False):
        if dummy:
            self.status = Status.DONE
            self.failures = None
            return

        self.failures = None
        if result is None:
            self.status = Status.RUNNING
            return

        self.stdouts = {k: open(s, "r").read() for k, s in result.stdouts.items()}
        self.stderrs = {k: open(s, "r").read() for k, s in result.stderrs.items()}

        if result.is_failed():
            self.status = Status.FAILED
            self.failures = result.failures
        else:
            self.status = Status.DONE

    def is_failed(self):
        return self.status == Status.FAILED

    def is_done(self):
        return self.status == Status.DONE

    def __repr__(self):
        return str(self.__dict__)


def launch(
    func: Callable,
    func_kwargs: dict[str, Any],
    #
    hostnames: list[str] = ["localhost"],
    workers_per_host: int | list[int] | None = 1,
    visible_devices_per_host: list[list[int]] | None = None,
    #
    ssh_config_file: str | os.PathLike | None = None,
    backend: Literal["mpi", "gloo", "nccl", "ucc"] | None = None,
    log_dir: str = "parallel_processing.log",  # TODO: use
):
    if not dist.is_available():
        raise RuntimeError("The torch.distributed package is not available.")

    # parse arguments

    num_nodes = len(hostnames)

    if workers_per_host is not None and isinstance(workers_per_host, int):
        workers_per_host = [workers_per_host] * num_nodes

    if visible_devices_per_host is not None:
        workers_per_host = [len(indices) for indices in visible_devices_per_host]

    assert workers_per_host is not None
    assert len(workers_per_host) == num_nodes

    launcher_hostname = socket.gethostname()
    launcher_ip = socket.gethostbyname(launcher_hostname)
    launcher_port = get_open_port()

    # start agents on each node
    for i, hostname in enumerate(hostnames):
        execute_ssh_command(
            command=f"{sys.executable} -u -m torchrunx {num_nodes+1} {i+1} {launcher_ip} {launcher_port}",
            hostname=hostname,
            ssh_config_file=ssh_config_file,
        )

    # initialize launcher–agent process group
    # ranks = (launcher, agent_0, ..., agent_{num_nodes-1})

    world_size = num_nodes + 1

    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=0,
        store=dist.TCPStore(launcher_hostname, launcher_port, is_master=True),  # pyright: ignore[reportPrivateImportUsage]
        timeout=timedelta(seconds=30),
    )

    # build LaunchConfig
    cumulative_workers = [0] + list(itertools.accumulate(workers_per_host))
    worker_global_ranks = [
        list(range(cumulative_workers[n], cumulative_workers[n + 1]))
        for n in range(num_nodes)
    ]

    config = LaunchConfig(
        fn=partial(func, **func_kwargs),
        world_size=cumulative_workers[-1],
        node_worker_ranks=worker_global_ranks,
        backend=backend
    )

    # broadcast agent parameters
    broadcast(object=config, src=0)  # LaunchConfig
    broadcast(object=(None, None), src=1)  # master_ip, master_port
    agent_pids: list[int] = gather(object=None, dst=0)  # pyright: ignore[reportAssignmentType]
    print(agent_pids)
    agent_pids = agent_pids[1:]

    # start monitoring loop
    dummy_launch_status = AgentStatus(None, True)
    while True:
        try:
            agent_statuses = all_gather(object=dummy_launch_status)
        except:
            # kill all agents (most should be dead but some could be hanging)
            for pid, ip_forgn in zip(agent_pids, hostnames):
                execute_ssh_command(
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
                        e += f"Node {i-1}, local worker {k} exited with error: {v.message['message']}\n"
                        e += f"{v.message['extraInfo']['py_callstack']}\n\n"
            raise RuntimeError(e)

        # else, check if everything's done
        if all(map(lambda s: s.is_done(), agent_statuses)):
            # we can exit loop and gather return values
            break

    # print stdouts and stderrs
    r = 0
    for node, status in enumerate(agent_statuses[1:]):
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
        outputs = gather(object={}, dst=0)
    except:
        for pid, ip_forgn in zip(agent_pids, hostnames):
            execute_ssh_command(
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

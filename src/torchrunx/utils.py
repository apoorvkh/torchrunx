from __future__ import annotations

import datetime
import io
import ipaddress
import os
import socket
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import cloudpickle
import fabric
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.api import RunProcsResult
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure
from typing_extensions import Self


def slurm_hosts() -> list[str]:
    """Retrieves hostnames of Slurm-allocated nodes.

    :return: Hostnames of nodes in current Slurm allocation
    :rtype: list[str]
    """
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

    :return: The implied number of workers per node.
    :rtype: int
    """
    # TODO: sanity check SLURM variables, commands
    assert "SLURM_JOB_ID" in os.environ
    if "SLURM_JOB_GPUS" in os.environ:
        # TODO: is it possible to allocate uneven GPUs across nodes?
        return len(os.environ["SLURM_JOB_GPUS"].split(","))
    else:
        # TODO: should we assume that we plan to do one worker per CPU?
        return int(os.environ["SLURM_CPUS_ON_NODE"])


def get_open_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


def is_localhost(hostname_or_ip: str) -> bool:
    # check if host is "loopback" address (i.e. designated to send to self)
    try:
        ip = ipaddress.ip_address(hostname_or_ip)
    except ValueError:
        ip = ipaddress.ip_address(socket.gethostbyname(hostname_or_ip))
    if ip.is_loopback:
        return True
    # else compare local interface addresses between host and localhost
    host_addrs = [addr[4][0] for addr in socket.getaddrinfo(str(ip), None)]
    localhost_addrs = [addr[4][0] for addr in socket.getaddrinfo(socket.gethostname(), None)]
    return len(set(host_addrs) & set(localhost_addrs)) > 0


def execute_command(
    command: str,
    hostname: str,
    ssh_config_file: str | os.PathLike | None = None,
    outfile: str | os.PathLike | None = None,
) -> None:
    # TODO: permit different stderr / stdout
    if is_localhost(hostname):
        _outfile = subprocess.DEVNULL
        if outfile is not None:
            _outfile = open(outfile, "w")
        subprocess.Popen(command, shell=True, stdout=_outfile, stderr=_outfile)
    else:
        with fabric.Connection(
            host=hostname, config=fabric.Config(runtime_ssh_path=ssh_config_file)
        ) as conn:
            if outfile is None:
                outfile = "/dev/null"
            conn.run(f"{command} >> {outfile} 2>&1 &", asynchronous=True)


@dataclass
class LauncherPayload:
    fn: Callable
    hostnames: list[str]
    worker_world_size: int
    worker_global_ranks: list[list[int]]
    worker_log_files: list[list[Path]]
    backend: Literal["mpi", "gloo", "nccl", "ucc", None]


@dataclass
class AgentPayload:
    hostname: str
    port: int
    process_id: int


@dataclass
class LauncherAgentGroup:
    launcher_hostname: str
    launcher_port: int
    world_size: int
    rank: int

    def __post_init__(self) -> None:
        self.group = dist.init_process_group(
            backend="gloo",
            world_size=self.world_size,
            rank=self.rank,
            store=dist.TCPStore(  # pyright: ignore[reportPrivateImportUsage]
                host_name=self.launcher_hostname,
                port=self.launcher_port,
                world_size=self.world_size,
                is_master=(self.rank == 0),
            ),
            timeout=datetime.timedelta(seconds=30),
        )

    def _serialize(self, object: Any) -> bytes:
        return cloudpickle.dumps(object)

    def _deserialize(self, serialized: bytes) -> Any:
        return cloudpickle.loads(serialized)

    def _all_gather(self, object: Any) -> list:
        """gather object from every rank to list on every rank"""
        object_bytes = self._serialize(object)
        object_list = [bytes()] * self.world_size
        dist.all_gather_object(object_list=object_list, obj=object_bytes, group=self.group)
        object_list = [self._deserialize(o) for o in object_list]
        return object_list

    def sync_payloads(
        self, payload: LauncherPayload | AgentPayload
    ) -> list[LauncherPayload | AgentPayload]:
        return self._all_gather(object=payload)

    def sync_agent_statuses(self, status: AgentStatus) -> list[AgentStatus]:
        return self._all_gather(object=status)[1:]


@dataclass
class AgentStatus:
    running: bool = True
    failed: bool = False
    return_values: dict[int, Any] = field(default_factory=dict)
    failures: dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: RunProcsResult | None, worker_global_ranks: list[int]) -> Self:
        if result is None:
            return cls()

        return cls(
            running=False,
            failed=result.is_failed(),
            return_values={worker_global_ranks[k]: v for k, v in result.return_values.items()},
            failures={worker_global_ranks[k]: v for k, v in result.failures.items()},
        )

    def is_running(self) -> bool:
        return self.running

    def is_failed(self) -> bool:
        return self.failed

    def is_done(self) -> bool:
        return not self.running and not self.failed


class WorkerTee(object):
    def __init__(self, name: os.PathLike | str, mode: str):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.__del__()

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


def monitor_log(log_file: Path):
    log_file.touch()
    f = open(log_file, "r")
    print(f.read())
    f.seek(0, io.SEEK_END)
    while True:
        new = f.read()
        if len(new) != 0:
            print(new)
        time.sleep(0.1)

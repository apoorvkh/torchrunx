from __future__ import annotations

import datetime
import ipaddress
import os
import socket
import subprocess
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import cloudpickle
import fabric
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.api import RunProcsResult
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure
from typing_extensions import Self


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
    command: str, hostname: str, ssh_config_file: str | os.PathLike | None = None
) -> None:
    if is_localhost(hostname):
        subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        with fabric.Connection(
            host=hostname, config=fabric.Config(runtime_ssh_path=ssh_config_file)
        ) as conn:
            conn.run(f"{command} >> /dev/null 2>&1 &", asynchronous=True)


@dataclass
class LauncherPayload:
    fn: Callable
    worker_world_size: int
    worker_global_ranks: list[list[int]]
    backend: Literal["mpi", "gloo", "nccl", "ucc", None]


@dataclass
class AgentPayload:
    ip: str
    port: int
    process_id: int


@dataclass
class LauncherAgentGroup:
    world_size: int
    rank: int
    launcher_hostname: str
    launcher_port: int

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
            stderrs={
                worker_global_ranks[k]: open(s, "r").read() for k, s in result.stderrs.items()
            },
            stdouts={
                worker_global_ranks[k]: open(s, "r").read() for k, s in result.stdouts.items()
            },
        )

    def is_running(self) -> bool:
        return self.running

    def is_failed(self) -> bool:
        return self.failed

    def is_done(self) -> bool:
        return not self.running and not self.failed

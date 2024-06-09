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


@dataclass
class LaunchConfig:
    fn: Callable
    world_size: int
    node_worker_ranks: list[list[int]]
    backend: Literal["mpi", "gloo", "nccl", "ucc", None]


@dataclass
class AgentStatus:
    running: bool = True
    failed: bool = False
    return_values: dict[int, Any] = field(default_factory=dict)
    failures: dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: RunProcsResult | None, worker_ranks: list[int]) -> Self:
        if result is not None:
            return cls(
                running=False,
                failed = result.is_failed(),
                return_values = {worker_ranks[k]: v for k, v in result.return_values.items()},
                failures = {worker_ranks[k]: v for k, v in result.failures.items()},
                stderrs = {worker_ranks[k]: open(s, "r").read() for k, s in result.stderrs.items()},
                stdouts = {worker_ranks[k]: open(s, "r").read() for k, s in result.stdouts.items()},
            )
        return cls()

    def is_running(self) -> bool:
        return self.running

    def is_failed(self) -> bool:
        return self.failed

    def is_done(self) -> bool:
        return not self.running and not self.failed


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
        subprocess.Popen(command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        with fabric.Connection(
            host=hostname, config=fabric.Config(runtime_ssh_path=ssh_config_file)
        ) as conn:
            conn.run(f"{command} >> /dev/null 2>&1 &")


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

    def _broadcast(
        self,
        object: Any,
        src: int = 0,
    ) -> Any:
        """broadcast object from src to all ranks"""
        data = [self._serialize(object)]
        dist.broadcast_object_list(object_list=data, src=src, group=self.group)
        return self._deserialize(data[0])

    def _gather(self, object: Any, dst: int = 0) -> list | None:
        """gather object from every rank to list in dst"""
        object_bytes = self._serialize(object)

        object_gather_list: list[bytes] | None = None
        if self.rank == dst:
            object_gather_list = [bytes()] * self.world_size

        dist.gather_object(
            obj=object_bytes,
            object_gather_list=object_gather_list,
            dst=dst,
            group=self.group,
        )

        if object_gather_list is None:
            return None

        return [self._deserialize(o) for o in object_gather_list]

    def _all_gather(self, object: Any) -> list:
        """gather object from every rank to list on every rank"""
        object_bytes = self._serialize(object)
        object_list = [bytes()] * self.world_size
        dist.all_gather_object(object_list=object_list, obj=object_bytes, group=self.group)
        object_list = [self._deserialize(o) for o in object_list]
        return object_list

    def send_launch_config(self, config: LaunchConfig) -> None:
        assert self.rank == 0
        self._broadcast(object=config, src=0)

    def recv_launch_config(self) -> LaunchConfig:
        assert self.rank > 0
        return self._broadcast(object=None, src=0)

    def send_process_id(self) -> None:
        assert self.rank > 0
        self._gather(object=os.getpid(), dst=0)

    def recv_agent_process_ids(self) -> list[int]:
        assert self.rank == 0
        agent_pids: list[int] = self._gather(object=None, dst=0)  # pyright: ignore[reportAssignmentType]
        return agent_pids[1:]

    def sync_main_agent_ip_port(self) -> tuple[str, int]:
        if self.rank == 1:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            port = get_open_port()
        else:
            ip, port = None, None
        return self._broadcast(object=(ip, port), src=1)

    def sync_agent_statuses(self, status: AgentStatus) -> list[AgentStatus]:
        return self._all_gather(object=status)[1:]

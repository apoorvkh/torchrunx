from __future__ import annotations

import datetime
import socket
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import cloudpickle
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.api import RunProcsResult
from typing_extensions import Self


def get_open_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


@dataclass
class WorkerException:
    exception: Exception


@dataclass
class LauncherPayload:
    fn: Callable
    hostnames: list[str]
    worker_world_size: int
    worker_global_ranks: list[list[int]]
    backend: Literal["mpi", "gloo", "nccl", "ucc", None]
    timeout: int


@dataclass
class AgentPayload:
    hostname: str
    port: int
    process_id: int


@dataclass
class AgentStatus:
    state: Literal["running", "failed", "done"]
    return_values: dict[int, Any | WorkerException] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: RunProcsResult | None, worker_global_ranks: list[int]) -> Self:
        if result is None:
            return cls(state="running")

        return_values = result.return_values

        if any(isinstance(v, WorkerException) for v in return_values.values()):
            state = "failed"
        else:
            state = "done"

        return cls(
            state=state,
            return_values={worker_global_ranks[k]: v for k, v in return_values.items()},
        )


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
        object_list = [b""] * self.world_size
        dist.all_gather_object(object_list=object_list, obj=object_bytes, group=self.group)
        object_list = [self._deserialize(o) for o in object_list]
        return object_list

    def sync_payloads(
        self, payload: LauncherPayload | AgentPayload
    ) -> tuple[LauncherPayload, list[AgentPayload]]:
        payloads = self._all_gather(object=payload)
        launcher_payload = payloads[0]
        agent_payloads = payloads[1:]
        return launcher_payload, agent_payloads

    def sync_agent_statuses(self, status: AgentStatus | None) -> list[AgentStatus]:
        return self._all_gather(object=status)[1:]  # [0] is launcher (status=None)

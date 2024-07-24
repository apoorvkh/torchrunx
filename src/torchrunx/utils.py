from __future__ import annotations

import datetime
import os
import socket
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import cloudpickle
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.api import RunProcsResult
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure
from typing_extensions import Self

from torchrunx.slurm import slurm_hosts, slurm_workers


def get_open_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


def automatic() -> tuple[list[str], int]:
    """
    Automatically determine allocation sizes

    :return: Allocation hosts and workers per host
    :rtype: tuple[list[str], int]
    """

    if "SLURM_JOB_ID" not in os.environ:
        _cpus = os.cpu_count()
        cpus = 1 if _cpus is None else _cpus
        gpus = torch.cuda.device_count()
        return (["localhost"], cpus if gpus == 0 else gpus)

    return slurm_hosts(), slurm_workers()


@dataclass
class LauncherPayload:
    fn: Callable
    hostnames: list[str]
    worker_world_size: int
    worker_global_ranks: list[list[int]]
    worker_log_files: list[list[Path]]
    backend: Literal["mpi", "gloo", "nccl", "ucc", None]
    timeout: int


@dataclass
class AgentPayload:
    hostname: str
    port: int
    process_id: int


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

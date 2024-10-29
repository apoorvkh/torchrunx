"""Utilities for Launcher-Agent communication."""

from __future__ import annotations

__all__ = [
    "get_open_port",
    "LauncherAgentGroup",
    "LauncherPayload",
    "AgentPayload",
    "ExceptionFromWorker",
    "AgentStatus",
]

import datetime
import socket
from contextlib import closing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

import cloudpickle
import torch.distributed as dist
from typing_extensions import Self

from .errors import AgentFailedError, ExceptionFromWorker, WorkerFailedError

if TYPE_CHECKING:
    from torch.distributed.elastic.multiprocessing.api import RunProcsResult


def get_open_port() -> int:
    """Return an open port number."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@dataclass
class LauncherAgentGroup:
    """Initializes a GLOO distributed process group between launcher and all agents."""

    launcher_hostname: str
    launcher_port: int
    world_size: int
    rank: int

    def __post_init__(self) -> None:
        """Initialize process group.

        Raises:
            torch.distributed.DistStoreError: if group initialization times out.
        """
        self.group = dist.init_process_group(
            backend="gloo",
            world_size=self.world_size,
            rank=self.rank,
            store=dist.TCPStore(  # pyright: ignore [reportPrivateImportUsage]
                host_name=self.launcher_hostname,
                port=self.launcher_port,
                world_size=self.world_size,
                is_master=(self.rank == 0),
            ),
            timeout=datetime.timedelta(seconds=30),
        )

    def _serialize(self, obj: Any) -> bytes:
        return cloudpickle.dumps(obj)

    def _deserialize(self, serialized: bytes) -> Any:
        return cloudpickle.loads(serialized)

    def _all_gather(self, obj: Any) -> list:
        """Gather object from every rank to list on every rank.

        Raises:
            AgentFailedError: if any agent fails (observed by this communication).
        """
        try:
            object_bytes = self._serialize(obj)
            object_list = [b""] * self.world_size
            # raises RuntimeError if timeout
            dist.all_gather_object(object_list=object_list, obj=object_bytes, group=self.group)
            return [self._deserialize(o) for o in object_list]
        except RuntimeError as e:
            # occurs if launcher or any agent dies and communication times out
            raise AgentFailedError from e

    def sync_payloads(
        self,
        payload: LauncherPayload | AgentPayload,
    ) -> tuple[LauncherPayload, list[AgentPayload]]:
        """All-gather payloads across launcher and all agents."""
        payloads = self._all_gather(payload)
        launcher_payload = payloads[0]
        agent_payloads = payloads[1:]
        return launcher_payload, agent_payloads

    def sync_agent_statuses(self, status: AgentStatus | None) -> list[AgentStatus]:
        """All-gather agent statuses across launcher and all agents."""
        return self._all_gather(status)[1:]  # [0] is launcher (status=None)

    def shutdown(self) -> None:
        """Terminate process group."""
        dist.destroy_process_group(group=self.group)


@dataclass
class LauncherPayload:
    """Payload from launcher to agents with runtime information."""

    fn: Callable
    hostnames: list[str]
    worker_global_ranks: list[list[int]]
    worker_world_size: int
    backend: Literal["nccl", "gloo", "mpi", "ucc", "auto"] | None
    timeout: int


@dataclass
class AgentPayload:
    """Payload corresponding to each agent."""

    hostname: str
    port: int
    process_id: int


@dataclass
class AgentStatus:
    """Status of each agent (to be synchronized in LauncherAgentGroup).

    Attributes:
        state: Whether the agent is running, failed, or done.
        return_values: Objects returned (or exceptions raised) by workers (indexed by local rank).
    """

    state: Literal["running", "failed", "done"]
    return_values: list[Any | WorkerFailedError | ExceptionFromWorker] = field(
        default_factory=list
    )  # indexed by local rank

    @classmethod
    def from_result(cls, result: RunProcsResult | None) -> Self:
        """Convert RunProcsResult (from polling worker process context) to AgentStatus."""
        if result is None:
            return cls(state="running")

        for local_rank, failure in result.failures.items():
            result.return_values[local_rank] = WorkerFailedError(failure.message)

        return_values = list(result.return_values.values())

        failed = any(isinstance(v, (ExceptionFromWorker, WorkerFailedError)) for v in return_values)
        state = "failed" if failed else "done"

        return cls(
            state=state,
            return_values=return_values,
        )

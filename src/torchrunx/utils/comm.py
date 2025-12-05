"""Utilities for Launcher-Agent communication."""

from __future__ import annotations

__all__ = [
    "AgentPayload",
    "AgentStatus",
    "ExceptionFromWorker",
    "LauncherAgentGroup",
    "LauncherPayload",
    "get_open_port",
]

import datetime
import socket
from contextlib import closing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import cloudpickle
import torch.distributed as dist

from .errors import AgentFailedError, ExceptionFromWorker, WorkerFailedError

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.elastic.multiprocessing.api import RunProcsResult


def get_open_port() -> int:
    """Return an open port number."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


ObjectT = TypeVar("ObjectT", bound=Any)
FunctionR = TypeVar("FunctionR")


@dataclass
class LauncherAgentGroup(Generic[FunctionR]):
    """Initializes a GLOO distributed process group between launcher and all agents."""

    launcher_hostname: str
    launcher_port: int
    world_size: int
    rank: int
    agent_timeout: int = 30

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
            timeout=datetime.timedelta(seconds=self.agent_timeout),
        )

    def _all_gather(self, obj: ObjectT) -> list[ObjectT]:
        """Gather object from each rank to list (in rank-order).

        Raises:
            AgentFailedError: if any agent fails (observed by this communication).
        """
        try:
            rank_obj = cloudpickle.dumps((self.rank, obj))
            all_gather_list = [b""] * self.world_size

            dist.all_gather_object(
                object_list=all_gather_list, obj=rank_obj, group=self.group
            )  # raises RuntimeError if timeout

            rank_obj_list: list[tuple[int, ObjectT]] = sorted(
                [cloudpickle.loads(o) for o in all_gather_list]
            )
            return [obj for _, obj in rank_obj_list]
        except RuntimeError as e:
            # occurs if launcher or any agent dies and communication times out
            raise AgentFailedError from e

    def sync_payloads(
        self,
        payload: LauncherPayload | AgentPayload,
    ) -> tuple[LauncherPayload, list[AgentPayload]]:
        """All-gather payloads across launcher and all agents."""
        payloads = self._all_gather(payload)
        launcher_payload: LauncherPayload = payloads[0]  # pyright: ignore [reportAssignmentType]
        agent_payloads: list[AgentPayload] = payloads[1:]  # pyright: ignore [reportAssignmentType]
        return launcher_payload, agent_payloads

    def sync_agent_statuses(
        self, status: AgentStatus[FunctionR] | None
    ) -> list[AgentStatus[FunctionR]]:
        """All-gather agent statuses across launcher and all agents."""
        # only launcher has status = None
        agent_statuses: list[AgentStatus[FunctionR]] = self._all_gather(status)[1:]  # pyright: ignore [reportAssignmentType]
        return agent_statuses

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
    backend: Literal["nccl", "gloo", "mpi", "ucc"] | None
    worker_timeout: int


@dataclass
class AgentPayload:
    """Payload corresponding to each agent."""

    hostname: str
    port: int
    process_id: int


@dataclass
class AgentStatus(Generic[FunctionR]):
    """Status of each agent (to be synchronized in LauncherAgentGroup).

    Attributes:
        state: Whether the agent is running, failed, or done.
        return_values: Objects returned (or exceptions raised) by workers (indexed by local rank).
    """

    state: Literal["running", "failed", "done"]
    return_values: list[FunctionR | WorkerFailedError | ExceptionFromWorker] = field(
        default_factory=list
    )  # indexed by local rank

    @classmethod
    def from_result(cls, result: RunProcsResult | None) -> AgentStatus[FunctionR]:
        """Convert RunProcsResult (from polling worker process context) to AgentStatus."""
        if result is None:
            return cls(state="running")

        for local_rank, failure in result.failures.items():
            result.return_values[local_rank] = WorkerFailedError(failure.message)

        return_values = [result.return_values[key] for key in sorted(result.return_values.keys())]

        failed = any(isinstance(v, ExceptionFromWorker | WorkerFailedError) for v in return_values)
        state = "failed" if failed else "done"

        return cls(
            state=state,
            return_values=return_values,
        )

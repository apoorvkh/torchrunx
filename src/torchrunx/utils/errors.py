"""Exception classes for agents and workers."""

from dataclasses import dataclass

__all__ = [
    "AgentFailedError",
    "WorkerFailedError",
    "ExceptionFromWorker",
]


class AgentFailedError(Exception):
    """Raised if agent fails (e.g. if signal received)."""


class WorkerFailedError(Exception):
    """Raised if a worker fails (e.g. if signal recieved or segmentation fault)."""


@dataclass
class ExceptionFromWorker:
    """Container for exceptions raised inside workers (from user script)."""

    exception: Exception

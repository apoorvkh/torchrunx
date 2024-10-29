from dataclasses import dataclass

__all__ = [
    "AgentFailedError",
    "WorkerFailedError",
    "ExceptionFromWorker",
]


class AgentFailedError(Exception):
    pass


class WorkerFailedError(Exception):
    pass


@dataclass
class ExceptionFromWorker:
    exception: Exception

import importlib.metadata

from .launcher import DEFAULT_ENV_VARS_FOR_COPY, Launcher, LaunchResult
from .utils.errors import AgentFailedError, WorkerFailedError

__version__ = importlib.metadata.version(__package__ or __name__)

__all__ = [  # noqa: RUF022
    "DEFAULT_ENV_VARS_FOR_COPY",
    "Launcher",
    "LaunchResult",
    "AgentFailedError",
    "WorkerFailedError",
]

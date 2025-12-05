"""Utilities for intercepting logs in worker processes and handling these in the Launcher."""

from __future__ import annotations

__all__ = [
    "RedirectHandler",
    "default_handlers",
    "file_handlers",
    "get_handler_filter",
]

import datetime
import logging
import os
from logging import LogRecord
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def get_handler_filter(
    hostname: str,
    local_rank: int | None,  # None indicates agent
    log_level: int = logging.NOTSET,
) -> Callable[[LogRecord], bool]:
    """Get an agent- or worker- specific filter to apply to :obj:`logging.Handler`."""
    return lambda record: (
        record.hostname == hostname  # pyright: ignore [reportAttributeAccessIssue]
        and record.local_rank == local_rank  # pyright: ignore [reportAttributeAccessIssue]
        and record.levelno >= log_level
    )


class RedirectHandler(logging.Handler):
    """For handling logs from hostname/rank with a corresponding logger in the launcher process."""

    def emit(self, record: LogRecord) -> None:
        """Handle log record using corresponding logger."""
        logger = logging.getLogger(record.name)
        if logger.isEnabledFor(record.levelno):
            logger.handle(record)


def file_handlers(
    hostnames: list[str],
    workers_per_host: list[int],
    log_dir: str | os.PathLike = Path("torchrunx_logs"),
    log_level: int = logging.NOTSET,
) -> list[logging.Handler]:
    """Handler builder function for writing logs for all workers/agents to a directory.

    Files are named with hostname and the local_rank (for workers).
    """
    handlers = []

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    log_dir = Path(log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    for hostname, num_workers in zip(hostnames, workers_per_host, strict=True):
        for local_rank in [None, *range(num_workers)]:
            local_rank_str = f"[{local_rank}]" if local_rank is not None else ""
            file_path = log_dir / f"{hostname}{local_rank_str}.log"

            h = logging.FileHandler(file_path)
            h.addFilter(get_handler_filter(hostname, local_rank, log_level=log_level))
            h.setFormatter(formatter)

            handlers.append(h)

    return handlers


def default_handlers(hostnames: list[str], workers_per_host: list[int]) -> list[logging.Handler]:
    """Constructs default :obj:`logging.Handler` objects.

    Logs for the rank 0 agent and rank 0 worker are redirected to loggers in the launcher process.
    Logs for all hosts/workers are written to files in ``$TORCHRUNX_LOG_DIR`` (named by timestamp,
    hostname, local_rank). If ``$TORCHRUNX_LOG_DIR = ""`` logs will not be written to files.
    """
    handlers = []

    # Agent 0 handler
    handlers.append(RedirectHandler())
    handlers[-1].addFilter(get_handler_filter(hostnames[0], None))

    # Worker 0 handler
    handlers.append(RedirectHandler())
    handlers[-1].addFilter(get_handler_filter(hostnames[0], 0))

    # Log to directory

    log_dir = os.environ.get("TORCHRUNX_LOG_DIR", "torchrunx_logs")
    if log_dir != "":
        file_log_level = os.environ.get("TORCHRUNX_LOG_LEVEL", "INFO")
        if file_log_level.isdigit():
            file_log_level = int(file_log_level)
        elif file_log_level in logging._nameToLevel:  # noqa: SLF001
            file_log_level = logging._nameToLevel[file_log_level]  # noqa: SLF001
        else:
            msg = (
                f"Invalid value for $TORCHRUNX_LOG_LEVEL: {file_log_level}. "
                f"Should be a positive integer or any of: {', '.join(logging._nameToLevel.keys())}."  # noqa: SLF001
            )
            raise ValueError(msg)

        handlers += file_handlers(
            hostnames, workers_per_host, log_dir=Path(log_dir), log_level=file_log_level
        )

    return handlers

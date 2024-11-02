"""Utilities for intercepting logs in worker processes and handling these in the Launcher."""

from __future__ import annotations

__all__ = [
    "LoggingServerArgs",
    "start_logging_server",
    "redirect_stdio_to_logger",
    "log_records_to_socket",
    "add_filter_to_handler",
    "file_handler",
    "stream_handler",
    "file_handlers",
    "default_handlers",
]

import datetime
import logging
import pickle
import signal
import struct
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from logging import Handler, Logger
from logging.handlers import SocketHandler
from pathlib import Path
from socketserver import StreamRequestHandler, ThreadingTCPServer
from typing import TYPE_CHECKING, Callable, Literal

import cloudpickle
from typing_extensions import Self

if TYPE_CHECKING:
    import os
    from multiprocessing.synchronize import Event as EventClass

## Handler utilities


def add_filter_to_handler(
    handler: Handler,
    hostname: str,
    local_rank: int | None,  # None indicates agent
    log_level: int = logging.NOTSET,
) -> None:
    """Apply a filter to :mod:`logging.Handler` so only specific worker logs are handled.

    Args:
        handler: Handler to be modified.
        hostname: Name of specified host.
        local_rank: Rank of specified worker on host (or ``None`` for agent itself).
        log_level: Minimum log level to capture.
    """

    def _filter(record: WorkerLogRecord) -> bool:
        return (
            record.hostname == hostname
            and record.local_rank == local_rank
            and record.levelno >= log_level
        )

    handler.addFilter(_filter)  # pyright: ignore [reportArgumentType]


def stream_handler(
    hostname: str, local_rank: int | None, log_level: int = logging.NOTSET
) -> Handler:
    """Handler builder function for writing logs from specified hostname/rank to stdout."""
    handler = logging.StreamHandler(stream=sys.stdout)
    add_filter_to_handler(handler, hostname, local_rank, log_level=log_level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s:%(levelname)s:%(hostname)s[%(local_rank)s]: %(message)s"
            if local_rank is not None
            else "%(asctime)s:%(levelname)s:%(hostname)s: %(message)s",
        ),
    )
    return handler


def file_handler(
    hostname: str,
    local_rank: int | None,
    file_path: str | os.PathLike,
    log_level: int = logging.NOTSET,
) -> Handler:
    """Handler builder function for writing logs from specified hostname/rank to a file."""
    handler = logging.FileHandler(file_path)
    add_filter_to_handler(handler, hostname, local_rank, log_level=log_level)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    return handler


def file_handlers(
    hostnames: list[str],
    workers_per_host: list[int],
    log_dir: str | os.PathLike = Path("torchrunx_logs"),
    log_level: int = logging.NOTSET,
) -> list[Handler]:
    """Handler builder function for writing logs for all workers/agents to a directory.

    Files are named with timestamp, hostname, and the local_rank (for workers).
    """
    handlers = []

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    for hostname, num_workers in zip(hostnames, workers_per_host):
        for local_rank in [None, *range(num_workers)]:
            file_path = (
                f"{log_dir}/{timestamp}-{hostname}"
                + (f"[{local_rank}]" if local_rank is not None else "")
                + ".log"
            )
            handlers.append(file_handler(hostname, local_rank, file_path, log_level=log_level))

    return handlers


def default_handlers(
    hostnames: list[str],
    workers_per_host: list[int],
    log_dir: str | os.PathLike = Path("torchrunx_logs"),
    log_level: int = logging.INFO,
) -> list[Handler]:
    """Default :mod:`logging.Handler`s for ``log_handlers="auto"`` in :mod:`torchrunx.launch`.

    Logs for ``host[0]`` and its ``local_rank[0]`` worker are written to launcher process stdout.
    Logs for all agents/workers are written to files in ``log_dir`` (named by timestamp, hostname,
    local_rank).
    """
    return [
        stream_handler(hostname=hostnames[0], local_rank=None, log_level=log_level),
        stream_handler(hostname=hostnames[0], local_rank=0, log_level=log_level),
        *file_handlers(hostnames, workers_per_host, log_dir=log_dir, log_level=log_level),
    ]


## Launcher utilities


class _LogRecordSocketReceiver(ThreadingTCPServer):
    """TCP server for recieving Agent/Worker log records in Launcher.

    Uses threading to avoid bottlenecks (i.e. "out-of-order" logs in Launcher process).
    """

    def __init__(self, host: str, port: int, handlers: list[Handler]) -> None:
        """Processing streamed bytes as LogRecord objects."""
        self.host = host
        self.port = port

        class _LogRecordStreamHandler(StreamRequestHandler):
            def handle(self) -> None:
                while True:
                    chunk_size = 4
                    chunk = self.connection.recv(chunk_size)
                    if len(chunk) < chunk_size:
                        break
                    slen = struct.unpack(">L", chunk)[0]
                    chunk = self.connection.recv(slen)
                    while len(chunk) < slen:
                        chunk = chunk + self.connection.recv(slen - len(chunk))
                    obj = pickle.loads(chunk)
                    record = logging.makeLogRecord(obj)

                    for handler in handlers:
                        handler.handle(record)

        super().__init__(
            server_address=(host, port),
            RequestHandlerClass=_LogRecordStreamHandler,
            bind_and_activate=True,
        )
        self.daemon_threads = True

    def shutdown(self) -> None:
        """Override BaseServer.shutdown() with added timeout (to avoid hanging)."""
        self._BaseServer__shutdown_request = True
        self._BaseServer__is_shut_down.wait(timeout=3)  # pyright: ignore[reportAttributeAccessIssue]


@dataclass
class LoggingServerArgs:
    """Arguments for starting a :class:`_LogRecordSocketReceiver`."""

    handler_factory: Callable[[], list[Handler]] | Literal["auto"] | None
    logging_hostname: str
    logging_port: int
    hostnames: list[str]
    workers_per_host: list[int]
    log_dir: str | os.PathLike
    log_level: int

    def serialize(self) -> bytes:
        """Serialize :class:`LoggingServerArgs` for passing to a new process."""
        return cloudpickle.dumps(self)

    @staticmethod
    def deserialize(serialized: bytes) -> LoggingServerArgs:
        """Deserialize bytes to :class:`LoggingServerArgs`."""
        return cloudpickle.loads(serialized)


def start_logging_server(
    serialized_args: bytes,
    stop_event: EventClass,
) -> None:
    """Serve :class:`_LogRecordSocketReceiver` until stop event triggered."""
    args = LoggingServerArgs.deserialize(serialized_args)

    log_handlers = []
    if args.handler_factory is None:
        log_handlers = []
    elif args.handler_factory == "auto":
        log_handlers = default_handlers(
            hostnames=args.hostnames,
            workers_per_host=args.workers_per_host,
            log_dir=args.log_dir,
            log_level=args.log_level,
        )
    elif isinstance(args.handler_factory, Callable):
        log_handlers = args.handler_factory()

    log_receiver = _LogRecordSocketReceiver(
        host=args.logging_hostname,
        port=args.logging_port,
        handlers=log_handlers,
    )

    try:
        log_receiver.serve_forever()
    except KeyboardInterrupt:
        sys.exit(128 + signal.SIGINT)

    while not stop_event.is_set():
        pass

    log_receiver.shutdown()
    log_receiver.server_close()


## Agent/worker utilities


def redirect_stdio_to_logger(logger: Logger) -> None:
    """Redirect stderr/stdout: send output to logger at every flush."""

    class _LoggingStream(StringIO):
        def __init__(self, logger: Logger, level: int = logging.NOTSET) -> None:
            super().__init__()
            self.logger = logger
            self.level = level

        def flush(self) -> None:
            super().flush()  # At "flush" to avoid logs of partial bytes
            value = self.getvalue()
            if value != "":
                self.logger.log(self.level, value)
                self.truncate(0)
                self.seek(0)

    logging.captureWarnings(capture=True)
    redirect_stderr(_LoggingStream(logger, level=logging.ERROR)).__enter__()
    redirect_stdout(_LoggingStream(logger, level=logging.INFO)).__enter__()


@dataclass
class WorkerLogRecord(logging.LogRecord):
    """Adding hostname, local_rank attributes to LogRecord. local_rank=None for Agent."""

    hostname: str
    local_rank: int | None

    @classmethod
    def from_record(cls, record: logging.LogRecord, hostname: str, local_rank: int | None) -> Self:
        record.hostname = hostname
        record.local_rank = local_rank
        record.__class__ = cls
        return record  # pyright: ignore [reportReturnType]


def log_records_to_socket(
    logger: Logger,
    hostname: str,
    local_rank: int | None,  # None indicates agent
    logger_hostname: str,
    logger_port: int,
) -> None:
    """Encode LogRecords with hostname/local_rank. Send to TCP socket on Launcher."""
    logger.setLevel(logging.NOTSET)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs) -> WorkerLogRecord:  # noqa: ANN002, ANN003
        record = old_factory(*args, **kwargs)
        return WorkerLogRecord.from_record(record, hostname, local_rank)

    logging.setLogRecordFactory(record_factory)

    logger.addHandler(SocketHandler(host=logger_hostname, port=logger_port))

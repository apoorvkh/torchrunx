"""Utilities for intercepting logs in worker processes and handling these in the Launcher."""

from __future__ import annotations

__all__ = [
    "LoggingServerArgs",
    "log_records_to_socket",
    "redirect_stdio_to_logger",
    "start_logging_server",
]

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
from multiprocessing.synchronize import Event as EventClass
from socketserver import StreamRequestHandler, ThreadingTCPServer
from typing import Callable

import cloudpickle
from typing_extensions import Self

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

                    ## Transform log record

                    record: WorkerLogRecord = logging.makeLogRecord(obj)  # pyright: ignore [reportAssignmentType]

                    if record.name != "root":
                        record.msg = f"{record.name}:{record.msg}"

                    record.name = f"torchrunx.{record.hostname}"
                    if record.local_rank is not None:
                        record.name += f".{record.local_rank}"

                    ## Handle log record

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

    handler_factory: Callable[[], list[Handler]]
    logging_hostname: str
    logging_port: int

    def serialize(self) -> bytes:
        """Serialize :class:`LoggingServerArgs` for passing to a new process."""
        return cloudpickle.dumps(self)

    @classmethod
    def from_bytes(cls, serialized: bytes) -> Self:
        """Deserialize bytes to :class:`LoggingServerArgs`."""
        return cloudpickle.loads(serialized)


def start_logging_server(serialized_args: bytes, stop_event: EventClass) -> None:
    """Serve :class:`_LogRecordSocketReceiver` until stop event triggered."""
    args = LoggingServerArgs.from_bytes(serialized_args)

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
    hostname: str,
    local_rank: int | None,  # None indicates agent
    logger_hostname: str,
    logger_port: int,
) -> None:
    """Encode LogRecords with hostname/local_rank. Send to TCP socket on Launcher."""
    logging.root.setLevel(logging.NOTSET)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs) -> WorkerLogRecord:  # noqa: ANN002, ANN003
        record = old_factory(*args, **kwargs)
        return WorkerLogRecord.from_record(record, hostname, local_rank)

    logging.setLogRecordFactory(record_factory)

    logging.root.addHandler(SocketHandler(host=logger_hostname, port=logger_port))

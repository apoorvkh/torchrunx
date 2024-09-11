from __future__ import annotations

import datetime
import logging
import os
import pickle
import struct
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from logging import Handler, Logger
from logging.handlers import SocketHandler
from pathlib import Path
from socketserver import StreamRequestHandler, ThreadingTCPServer

## Handler utilities


def add_filter_to_handler(
    handler: Handler,
    hostname: str,
    worker_rank: int | None,
    log_level: int = logging.NOTSET,
) -> None:
    def _filter(record: logging.LogRecord) -> bool:
        return (
            record.hostname == hostname  # pyright: ignore[reportAttributeAccessIssue]
            and record.worker_rank == worker_rank  # pyright: ignore[reportAttributeAccessIssue]
            and record.levelno >= log_level
        )

    handler.addFilter(_filter)


def file_handler(
    hostname: str,
    worker_rank: int | None,
    file_path: str | os.PathLike,
    log_level: int = logging.NOTSET,
) -> Handler:
    handler = logging.FileHandler(file_path)
    add_filter_to_handler(handler, hostname, worker_rank, log_level=log_level)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    return handler


def file_handlers(
    hostnames: list[str],
    workers_per_host: list[int],
    log_dir: str | os.PathLike = Path("torchrunx_logs"),
    log_level: int = logging.NOTSET,
) -> list[Handler]:
    handlers = []

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    for hostname, num_workers in zip(hostnames, workers_per_host):
        for rank in [None] + list(range(num_workers)):
            file_path = (
                f"{log_dir}/{timestamp}-{hostname}"
                + (f"[{rank}]" if rank is not None else "")
                + ".log"
            )
            handlers.append(file_handler(hostname, rank, file_path, log_level=log_level))

    return handlers


def stream_handler(hostname: str, rank: int | None, log_level: int = logging.NOTSET) -> Handler:
    handler = logging.StreamHandler()
    add_filter_to_handler(handler, hostname, rank, log_level=log_level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s:%(levelname)s:%(hostname)s[%(worker_rank)s]: %(message)s"
            if rank is not None
            else "%(asctime)s:%(levelname)s:%(hostname)s: %(message)s"
        )
    )
    return handler


def default_handlers(
    hostnames: list[str],
    workers_per_host: list[int],
    log_dir: str | os.PathLike = Path("torchrunx_logs"),
    log_level: int = logging.INFO,
) -> list[Handler]:
    return [
        stream_handler(hostname=hostnames[0], rank=None, log_level=log_level),
        stream_handler(hostname=hostnames[0], rank=0, log_level=log_level),
    ] + file_handlers(hostnames, workers_per_host, log_dir=log_dir, log_level=log_level)


## Agent/worker utilities


def log_records_to_socket(
    logger: Logger,
    hostname: str,
    worker_rank: int | None,
    logger_hostname: str,
    logger_port: int,
) -> None:
    logger.setLevel(logging.NOTSET)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs) -> logging.LogRecord:  # noqa: ANN002, ANN003
        record = old_factory(*args, **kwargs)
        record.hostname = hostname
        record.worker_rank = worker_rank
        return record

    logging.setLogRecordFactory(record_factory)

    logger.addHandler(SocketHandler(host=logger_hostname, port=logger_port))


def redirect_stdio_to_logger(logger: Logger) -> None:
    class _LoggingStream(StringIO):
        def __init__(self, logger: Logger, level: int = logging.NOTSET) -> None:
            super().__init__()
            self.logger = logger
            self.level = level

        def flush(self) -> None:
            super().flush()
            value = self.getvalue()
            if value != "":
                self.logger.log(self.level, f"\n{value}")
                self.truncate(0)
                self.seek(0)

    logging.captureWarnings(True)
    redirect_stderr(_LoggingStream(logger, level=logging.ERROR)).__enter__()
    redirect_stdout(_LoggingStream(logger, level=logging.INFO)).__enter__()


## Launcher utilities


class LogRecordSocketReceiver(ThreadingTCPServer):
    def __init__(self, host: str, port: int, handlers: list[Handler]) -> None:
        self.host = host
        self.port = port

        class _LogRecordStreamHandler(StreamRequestHandler):
            def handle(self) -> None:
                while True:
                    chunk = self.connection.recv(4)
                    if len(chunk) < 4:
                        break
                    slen = struct.unpack(">L", chunk)[0]
                    chunk = self.connection.recv(slen)
                    while len(chunk) < slen:
                        chunk = chunk + self.connection.recv(slen - len(chunk))
                    obj = pickle.loads(chunk)
                    record = logging.makeLogRecord(obj)
                    #
                    for handler in handlers:
                        handler.handle(record)

        super().__init__(
            server_address=(host, port),
            RequestHandlerClass=_LogRecordStreamHandler,
            bind_and_activate=True,
        )
        self.daemon_threads = True

    def shutdown(self) -> None:
        """override BaseServer.shutdown() with added timeout"""
        self._BaseServer__shutdown_request = True
        self._BaseServer__is_shut_down.wait(timeout=3)  # pyright: ignore[reportAttributeAccessIssue]

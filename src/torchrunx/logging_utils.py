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
from socketserver import StreamRequestHandler, TCPServer


class WorkerLogFilter(logging.Filter):
    def __init__(self, hostname: str, worker_rank: int | None):
        self.hostname = hostname
        self.worker_rank = worker_rank

    def filter(self, record: logging.LogRecord) -> bool:
        return record.hostname == self.hostname and record.worker_rank == self.worker_rank


def file_handlers(hostnames: list[str], workers_per_host: list[int]) -> list[Handler]:
    handlers = []

    log_dir = os.environ.get("TORCHRUNX_DIR", "./torchrunx_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    for hostname, num_workers in zip(hostnames, workers_per_host):
        for rank in [None] + list(range(num_workers)):
            handler = logging.FileHandler(
                f"{log_dir}/{timestamp}-{hostname}{'' if rank is None else f'[{rank}]'}.log"
            )
            formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")

            handler.addFilter(WorkerLogFilter(hostname, rank))
            handler.setLevel(logging.NOTSET)
            handler.setFormatter(formatter)

            handlers.append(handler)

    return handlers


def stream_handler(hostname: str, rank: int | None) -> Handler:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(hostname)s"
        + ("[%(worker_rank)s]" if rank is not None else "")
        + ": %(message)s"
    )
    handler.addFilter(WorkerLogFilter(hostname, rank))
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(formatter)
    return handler


def default_handlers(hostnames: list[str], workers_per_host: list[int]) -> list[Handler]:
    stream_handlers = [
        stream_handler(hostname=hostnames[0], rank=None),
        stream_handler(hostname=hostnames[0], rank=0),
    ]
    return stream_handlers + file_handlers(hostnames, workers_per_host)


## Agent/worker utilities


def log_records_to_socket(
    logger: Logger,
    hostname: str,
    worker_rank: int | None,
    logger_hostname: str,
    logger_port: int,
):
    logger.setLevel(logging.NOTSET)

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.hostname = hostname
        record.worker_rank = worker_rank
        return record

    logging.setLogRecordFactory(record_factory)

    logger.addHandler(SocketHandler(host=logger_hostname, port=logger_port))


def redirect_stdio_to_logger(logger: Logger):
    logging.captureWarnings(True)
    redirect_stderr(LoggingStream(logger, level=logging.ERROR)).__enter__()
    redirect_stdout(LoggingStream(logger, level=logging.INFO)).__enter__()


class LoggingStream(StringIO):
    def __init__(self, logger: Logger, level: int = logging.NOTSET):
        super().__init__()
        self.logger = logger
        self.level = level

    def flush(self):
        super().flush()
        value = self.getvalue()
        if value != "":
            self.logger.log(self.level, f"\n{value}")
            self.truncate(0)
            self.seek(0)


## Launcher utilities


class LogRecordSocketReceiver(TCPServer):
    def __init__(self, host: str, port: int, handlers: list[Handler]):
        class _LogRecordStreamHandler(StreamRequestHandler):
            def handle(self):
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
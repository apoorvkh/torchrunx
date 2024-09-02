from __future__ import annotations

import datetime
import logging
import os
import pickle
import struct
from io import StringIO
from logging import Handler, Logger
from logging.handlers import SocketHandler
from socketserver import StreamRequestHandler, ThreadingTCPServer


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


def default_handlers(hostnames: list[str], workers_per_host: list[int]) -> list[Handler]:
    handlers = []

    log_dir = os.environ.get("TORCHRUNX_DIR", "./torchrunx_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    def make_handler(hostname: str, rank: int | None = None, stream: bool = False) -> Handler:
        if stream:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s:%(levelname)s:%(hostname)s"
                + ("[%(worker_rank)s]" if rank is not None else "")
                + ": %(message)s"
            )
        else:
            handler = logging.FileHandler(
                f"{log_dir}/{timestamp}-{hostname}{'' if rank is None else f'[{rank}]'}.log"
            )
            formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")

        def handler_filter(record: logging.LogRecord) -> bool:
            return record.hostname == hostname and record.worker_rank == rank  # pyright: ignore

        handler.addFilter(handler_filter)
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(formatter)

        return handler

    for r in [None, 0]:
        handlers.append(make_handler(hostname=hostnames[0], rank=r, stream=True))

    for hostname, num_workers in zip(hostnames, workers_per_host):
        for j in [None] + list(range(num_workers)):
            handlers.append(make_handler(hostname=hostname, rank=j))

    return handlers


class LogRecordSocketReceiver(ThreadingTCPServer):
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

        super().__init__((host, port), _LogRecordStreamHandler)
        self.daemon_threads = True


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

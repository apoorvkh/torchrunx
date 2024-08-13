from __future__ import annotations

import datetime
import logging
import logging.handlers
import pickle
import select
import socketserver
import struct
from abc import ABC, abstractmethod
from collections import defaultdict
from io import StringIO, TextIOWrapper
from pathlib import Path


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """

    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            slen = struct.unpack(">L", chunk)[0]
            chunk = self.connection.recv(slen)
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        if self.server.logname is not None:  # type: ignore
            name = self.server.logname  # type: ignore
        else:
            name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        if logger.getEffectiveLevel() <= record.levelno:
            logger.handle(record)


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver suitable for testing.
    """

    allow_reuse_address = 1  # type: ignore

    def __init__(
        self,
        host="localhost",
        port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        handler=LogRecordStreamHandler,
    ):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = 0
        self.timeout = 1
        self.logname = None

    def serve_until_stopped(self):
        abort = 0
        while not abort:
            rd, wr, ex = select.select([self.socket.fileno()], [], [], self.timeout)
            if rd:
                self.handle_request()
            abort = self.abort


class RenamingSocketHandler(logging.handlers.SocketHandler):
    def __init__(self, host, port, root_name):
        super().__init__(host, port)

        self.root_name = root_name

    def emit(self, record):
        if not record.name.startswith(self.root_name):
            record.name = f"{self.root_name}.{record.name}"
        super().emit(record)


class LogSpec(ABC):
    @abstractmethod
    def get_map(self) -> dict[str, list[logging.Handler]]:
        """
        Called by torchrunx.launch on the log_spec argument.
        """
        raise NotImplementedError


class DefaultLogSpec(LogSpec):
    def __init__(self, log_spec_dict: dict[str, list[logging.Handler]]):
        self.log_spec_dict = log_spec_dict

    @classmethod
    def basic(
        cls, hostnames: list[str], num_workers: int, log_dir: str = "./logs", stream: bool = True
    ) -> DefaultLogSpec:
        """
        Generates torchrunx's default LogSpec

        :param hostnames: The node hostnames
        :type hostnames: list[str]
        :param num_agents: Number of agents in work group
        :type num_agents: int
        :param num_workers: Number of workers per agent
        :type num_workers: int
        :return: A logging structure to be passed to :mod:`torchrunx.launch` as the ``log_spec`` argument
        :rtype: dict[str, list[logging.Handler]]
        """  # noqa: E501

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        agents: dict[str, list[logging.Handler]] = {
            hostname: [logging.FileHandler(f"{log_dir}/{timestamp}-{hostname}.log")]
            for hostname in hostnames
        }
        workers: dict[str, list[logging.Handler]] = {
            f"{hostname}[{j}]": [logging.FileHandler(f"{log_dir}/{timestamp}-{hostname}[{j}].log")]
            for j in range(num_workers)
            for hostname in hostnames
        }

        if stream:
            workers[f"{hostnames[0]}[0]"].append(logging.StreamHandler())

        return cls({**agents, **workers})

    @classmethod
    def from_file_map(
        cls, file_map: dict[str, list[str]], log_dir: str = "./logs"
    ) -> DefaultLogSpec:
        """
        Generates DefaultLogSpec from a mapping of filenames to worker/agent names that should be logged there.

        :param file_map: A dictionary mapping file suffixes (filenames will be prefixed with a timestamp) to worker and agent names.
        :type file_map: dict[str, str]
        :return: Returns an accordingly constructed DefaultLogSpec
        :rtype: DefaultLogSpec
        """  # noqa: E501

        reverse_map: defaultdict[str, list[logging.Handler]] = defaultdict(lambda: [])

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        Path(log_dir).mkdir(parents=True, exist_ok=True)

        for file_suffix, loggers in file_map.items():
            for logger in loggers:
                reverse_map[logger].append(
                    logging.FileHandler(f"{log_dir}/{timestamp}-{file_suffix}")
                )

        return DefaultLogSpec(reverse_map)  # re-typing

    def get_map(self):
        return self.log_spec_dict


class StreamLogger:
    """
    For logging write calls to streams such as stdout and stdin in the worker processes.
    """
    def __init__(self, logger: logging.Logger, stream: TextIOWrapper | None):
        self.logger = logger
        self._string_io = StringIO()
        if stream is None:
            raise ValueError("stream cannot be None")
        self.stream: TextIOWrapper = stream  # type: ignore

    def write(self, data: str):
        self._string_io.write(data)
        self.stream.write(data)

    def flush(self):
        value = self._string_io.getvalue()
        if value != "":
            self.logger.info(f"\n{value}")
            self._string_io = StringIO()  # "create a new one, it's faster" - someone online
        self.stream.flush()

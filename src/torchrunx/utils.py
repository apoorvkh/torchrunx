from __future__ import annotations

import datetime
import logging
import logging.handlers
import pickle
import select
import socket
import socketserver
import struct
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import cloudpickle
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.api import RunProcsResult
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure
from typing_extensions import Self


def get_open_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


@dataclass
class LauncherPayload:
    fn: Callable
    hostnames: list[str]
    worker_world_size: int
    worker_global_ranks: list[list[int]]
    worker_log_names: list[list[str]]
    log_host: str
    backend: Literal["mpi", "gloo", "nccl", "ucc", None]
    timeout: int


@dataclass
class AgentPayload:
    hostname: str
    port: int
    process_id: int


@dataclass
class AgentStatus:
    running: bool = True
    failed: bool = False
    return_values: dict[int, Any] = field(default_factory=dict)
    failures: dict[int, ProcessFailure] = field(default_factory=dict)
    stdouts: dict[int, str] = field(default_factory=dict)
    stderrs: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: RunProcsResult | None, worker_global_ranks: list[int]) -> Self:
        if result is None:
            return cls()

        return cls(
            running=False,
            failed=result.is_failed(),
            return_values={worker_global_ranks[k]: v for k, v in result.return_values.items()},
            failures={worker_global_ranks[k]: v for k, v in result.failures.items()},
        )

    def is_running(self) -> bool:
        return self.running

    def is_failed(self) -> bool:
        return self.failed

    def is_done(self) -> bool:
        return not self.running and not self.failed


@dataclass
class LauncherAgentGroup:
    launcher_hostname: str
    launcher_port: int
    world_size: int
    rank: int

    def __post_init__(self) -> None:
        self.group = dist.init_process_group(
            backend="gloo",
            world_size=self.world_size,
            rank=self.rank,
            store=dist.TCPStore(  # pyright: ignore[reportPrivateImportUsage]
                host_name=self.launcher_hostname,
                port=self.launcher_port,
                world_size=self.world_size,
                is_master=(self.rank == 0),
            ),
            timeout=datetime.timedelta(seconds=30),
        )

    def _serialize(self, object: Any) -> bytes:
        return cloudpickle.dumps(object)

    def _deserialize(self, serialized: bytes) -> Any:
        return cloudpickle.loads(serialized)

    def _all_gather(self, object: Any) -> list:
        """gather object from every rank to list on every rank"""
        object_bytes = self._serialize(object)
        object_list = [bytes()] * self.world_size
        dist.all_gather_object(object_list=object_list, obj=object_bytes, group=self.group)
        object_list = [self._deserialize(o) for o in object_list]
        return object_list

    def sync_payloads(
        self, payload: LauncherPayload | AgentPayload
    ) -> list[LauncherPayload | AgentPayload]:
        return self._all_gather(object=payload)

    def sync_agent_statuses(self, status: AgentStatus) -> list[AgentStatus]:
        return self._all_gather(object=status)[1:]


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


def default_logging(
    hostnames: list[str], num_workers: int, log_dir: str,
    stream: bool = True
) -> dict[str, list[logging.Handler]]:
    """
    Generates torchrunx's default

    :param num_agents: Number of agents in work group
    :type num_agents: int
    :param num_workers: Number of workers per agent
    :type num_workers: int
    :return: A logging structure to be passed to :mod:`torchrunx.launch` as the ``log_spec`` argument
    :rtype: dict[str, list[logging.Handler]]
    """  # noqa: E501

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    agents: dict[str, list[logging.Handler]] = {
        hostname: [logging.FileHandler(f"{log_dir}/{timestamp}-{hostname}.log")]
        for hostname in hostnames
    }
    workers: dict[str, list[logging.Handler]] = {
        f"{hostname}.worker-{j}": [
            logging.FileHandler(f"{log_dir}/{timestamp}-{hostname}.worker-{j}.log")
        ]
        for j in range(num_workers)
        for hostname in hostnames
    }

    if stream:
        workers[f"{hostnames[0]}.worker-0"].append(logging.StreamHandler())

    return {**agents, **workers}

class RenamingSocketHandler(logging.handlers.SocketHandler):
    def __init__(self, host, port, root_name):
        super().__init__(host, port)

        self.root_name = root_name

    def emit(self, record):
        if not record.name.startswith(self.root_name):
            record.name = f"{self.root_name}.{record.name}"
        super().emit(record)

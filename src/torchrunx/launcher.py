from __future__ import annotations

import fnmatch
import ipaddress
import itertools
import logging
import logging.config
import logging.handlers
import os
import socket
import subprocess
import sys
from collections import ChainMap
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Process
from typing import Any, Callable, Literal

import fabric
import torch.distributed as dist

from .log_utils import DefaultLogSpec, LogRecordSocketReceiver, LogSpec
from .utils import (
    AgentPayload,
    AgentStatus,
    LauncherAgentGroup,
    LauncherPayload,
    get_open_port,
)


def is_localhost(hostname_or_ip: str) -> bool:
    # check if host is "loopback" address (i.e. designated to send to self)
    try:
        ip = ipaddress.ip_address(hostname_or_ip)
    except ValueError:
        ip = ipaddress.ip_address(socket.gethostbyname(hostname_or_ip))
    if ip.is_loopback:
        return True
    # else compare local interface addresses between host and localhost
    host_addrs = [addr[4][0] for addr in socket.getaddrinfo(str(ip), None)]
    localhost_addrs = [addr[4][0] for addr in socket.getaddrinfo(socket.gethostname(), None)]
    return len(set(host_addrs) & set(localhost_addrs)) > 0


def execute_command(
    command: str,
    hostname: str,
    ssh_config_file: str | os.PathLike | None = None,
) -> None:
    # TODO: permit different stderr / stdout
    if is_localhost(hostname):
        subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        with fabric.Connection(
            host=hostname, config=fabric.Config(runtime_ssh_path=ssh_config_file)
        ) as conn:
            conn.run(f"{command} >> /dev/null 2>&1 &", asynchronous=True)


def monitor_log(log_spec: LogSpec, port: int):
    log_formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s:%(message)s")
    for lname, handlers in log_spec.get_map().items():  # type: ignore
        _logger = logging.getLogger(f"torchrunx.{lname}")
        for handler in handlers:
            handler.setFormatter(log_formatter)
            _logger.addHandler(handler)

    LogRecordSocketReceiver(host=socket.getfqdn(), port=port).serve_until_stopped()


@dataclass
class Launcher:
    hostnames: list[str] = field(default_factory=lambda: ["localhost"])
    workers_per_host: int | list[int] = 1
    ssh_config_file: str | os.PathLike | None = None
    backend: Literal["mpi", "gloo", "nccl", "ucc", None] = None
    log_spec: LogSpec | None = None
    env_vars: list[str] = field(
        default_factory=lambda: [
            "PATH",
            "LD_LIBRARY",
            "LIBRARY_PATH",
            "PYTHON*",
            "CUDA*",
            "TORCH*",
            "PYTORCH*",
            "NCCL*",
        ]
    )
    env_file: str | os.PathLike | None = None
    timeout: int = 600

    def run(
        self,
        func: Callable,
        func_kwargs: dict[str, Any],
    ) -> dict[int, Any]:
        """
        Launch a distributed PyTorch function on the specified nodes. See :mod:`torchrunx.launch`

        :param func: The distributed function to call on all workers
        :type func: Callable
        :param func_kwargs: Any keyword arguments to be provided when calling ``func``
        :type func_kwargs: dict[str, Any]
        :raises RuntimeError: May fail due to misconfiguration, or errors thrown by ``func``
        :return: A dictionary mapping worker ranks to their output
        :rtype: dict[int, Any]
        """

        logger = logging.getLogger("torchrunx")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # log_dir = Path(self.log_dir)
        # log_dir.mkdir(parents=True, exist_ok=True)
        # timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        if self.log_spec is None:
            # TODO: this assumes the type of workers_per_host is simply int. We should consider
            # again whether it's worth supporting inhomogeneous allocations (list[int])
            self.log_spec = DefaultLogSpec.basic(
                hostnames=self.hostnames, num_workers=self.workers_per_host # type: ignore
            )  

        log_port = get_open_port()
        log_process = Process(target=monitor_log, args=(self.log_spec, log_port), daemon=True)
        log_process.start()

        if not dist.is_available():
            raise RuntimeError("The torch.distributed package is not available.")

        num_hosts = len(self.hostnames)

        workers_per_host = self.workers_per_host
        if isinstance(self.workers_per_host, int):
            workers_per_host = [workers_per_host] * num_hosts

        assert workers_per_host is not None
        assert len(workers_per_host) == num_hosts  # type: ignore

        # launch command

        env_exports = []
        for k, v in os.environ.items():
            if any(fnmatch.fnmatch(k, e) for e in self.env_vars):
                env_exports.append(f"{k}={v}")

        env_export_string = ""
        if len(env_exports) > 0:
            env_export_string = f"export {' '.join(env_exports)} && "

        env_file_string = ""
        if self.env_file is not None:
            env_file_string = f"source {self.env_file} && "

        launcher_hostname = socket.getfqdn()
        launcher_port = get_open_port()
        world_size = num_hosts + 1  # launcher + agents

        command = (
            f"cd {os.getcwd()} && "
            f"{env_export_string}"
            f"{env_file_string}"
            f"{sys.executable} -u -m torchrunx "
            f"--launcher-hostname {launcher_hostname} "
            f"--launcher-port {launcher_port} "
            f"--world-size {world_size} "
            # rank set in the loop below
        )

        logger.debug("starting agents")
        # start agents on each node
        for i, hostname in enumerate(self.hostnames):
            execute_command(
                command=f"{command} --rank {i+1}",
                hostname=hostname,
                ssh_config_file=self.ssh_config_file,
            )

        # initialize launcher–agent process group
        # ranks = (launcher, agent_0, ..., agent_{num_hosts-1})

        launcher_agent_group = LauncherAgentGroup(
            launcher_hostname=launcher_hostname,
            launcher_port=launcher_port,
            world_size=world_size,
            rank=0,
        )

        # build and sync payloads between launcher and agents

        _cumulative_workers = [0] + list(itertools.accumulate(workers_per_host))  # type: ignore

        worker_world_size = _cumulative_workers[-1]

        worker_global_ranks = []  # list of worker ranks per host
        for n in range(num_hosts):
            host_ranks = range(_cumulative_workers[n], _cumulative_workers[n + 1])
            worker_global_ranks.append(list(host_ranks))

        payload = LauncherPayload(
            fn=partial(func, **func_kwargs),
            hostnames=self.hostnames,
            worker_world_size=worker_world_size,
            worker_global_ranks=worker_global_ranks,
            log_host=launcher_hostname,
            log_port=log_port,
            backend=self.backend,
            timeout=self.timeout,
        )

        agent_payloads: list[AgentPayload] = launcher_agent_group.sync_payloads(payload=payload)[1:]  # pyright: ignore[reportAssignmentType]
        agent_pids = [p.process_id for p in agent_payloads]

        # loop to monitor agent statuses (until failed or done)
        try:
            while True:
                agent_statuses = launcher_agent_group.sync_agent_statuses(status=AgentStatus())

                if all(s.is_done() for s in agent_statuses):
                    break

                if any(s.is_failed() for s in agent_statuses):
                    # TODO: cleaner way to print these?
                    e = ""
                    for i, s in enumerate(agent_statuses):
                        if s is not None and s.is_failed():
                            for k, v in s.failures.items():
                                e += f"Node {i}, local worker {k} exited with error: "
                                if isinstance(v.message, str):
                                    e += f"{v.message}\n"
                                else:
                                    e += f"{v.message['message']}\n"
                                    e += f"{v.message['extraInfo']['py_callstack']}\n\n"
                    logger.error(f"workers threw:\n {e}")
                    raise RuntimeError(e)
        except:
            # cleanup: SIGTERM all agents
            logger.warn("agents encountered error, manually killing")
            for agent_pid, agent_hostname in zip(agent_pids, self.hostnames):
                execute_command(
                    command=f"kill {agent_pid}",
                    hostname=agent_hostname,
                    ssh_config_file=self.ssh_config_file,
                )
            raise
        finally:
            logger.debug("killing log process")
            log_process.kill()

        logger.debug("returning")
        return_values: dict[int, Any] = dict(ChainMap(*[s.return_values for s in agent_statuses]))
        return return_values


def launch(
    func: Callable,
    func_kwargs: dict[str, Any],
    hostnames: list[str] = ["localhost"],
    workers_per_host: int | list[int] = 1,
    ssh_config_file: str | os.PathLike | None = None,
    backend: Literal["mpi", "gloo", "nccl", "ucc", None] = None,
    log_dir: os.PathLike | str = "./logs",
    log_spec: LogSpec | None = None,
    env_vars: list[str] = [
        "PATH",
        "LD_LIBRARY",
        "LIBRARY_PATH",
        "PYTHON*",
        "CUDA*",
        "TORCH*",
        "PYTORCH*",
        "NCCL*",
    ],
    env_file: str | os.PathLike | None = None,
    timeout: int = 600,
) -> dict[int, Any]:
    """
    Launch a distributed PyTorch function on the specified nodes.

    :param func: The distributed function to call on all workers
    :type func: Callable
    :param func_kwargs: Any keyword arguments to be provided when calling ``func``
    :type func_kwargs: dict[str, Any]
    :param hostnames: A list of node hostnames to start workers on, defaults to ["localhost"]
    :type hostnames: list[str], optional
    :param workers_per_host: The number of workers per node. Providing an ``int`` implies all nodes should have ``workers_per_host`` workers, meanwhile providing a list causes node ``i`` to have ``worker_per_host[i]`` workers, defaults to 1
    :type workers_per_host: int | list[int], optional
    :param ssh_config_file: An SSH configuration file to use when connecting to nodes, defaults to None
    :type ssh_config_file: str | os.PathLike | None, optional
    :param backend: A ``torch.distributed`` `backend string <https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend>`_, defaults to None
    :type backend: Literal['mpi', 'gloo', 'nccl', 'ucc', None], optional
    :param log_dir: A directory in which logs should be written, defaults to "./logs"
    :type log_dir: os.PathLike | str, optional
    :param log_spec: TODO
    :type log_spec: TODO
    :param env_vars: A list of environmental variables to be copied from the launcher environment to workers. Allows for bash pattern matching syntax, defaults to ["PATH", "LD_LIBRARY", "LIBRARY_PATH", "PYTHON*", "CUDA*", "TORCH*", "PYTORCH*", "NCCL*"]
    :type env_vars: list[str], optional
    :param env_file: An additional environment file that will be sourced prior to executing ``func``, defaults to None
    :type env_file: str | os.PathLike | None, optional
    :param timeout: Worker process group timeout, defaults to 600
    :type timeout: int, optional
    :raises RuntimeError: May fail due to misconfiguration, or errors thrown by ``func``
    :return: A dictionary mapping worker ranks to their output
    :rtype: dict[int, Any]
    """  # noqa: E501
    return Launcher(
        hostnames=hostnames,
        workers_per_host=workers_per_host,
        ssh_config_file=ssh_config_file,
        backend=backend,
        log_spec=log_spec,
        env_vars=env_vars,
        env_file=env_file,
        timeout=timeout,
    ).run(func=func, func_kwargs=func_kwargs)

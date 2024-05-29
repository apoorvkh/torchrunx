from __future__ import annotations

import os, sys
import socket
from functools import partial
from typing import Callable, List, Tuple
from enum import Enum

from torchrunx.utils import get_open_port

import dill
import paramiko

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.api import RunProcsResult

class LaunchConfig:

    def __init__(self: LaunchConfig, fn: Callable, num_nodes: int, num_processes: int, backend: str) -> None:
        self.serialized_fn = dill.dumps(fn)
        self.num_nodes = num_nodes
        self.num_processes = num_processes
        self.backend = backend

    def serialize(self: LaunchConfig) -> bytes:
        return dill.dumps(self)

    @staticmethod
    def deserialize(serialized_config: bytes) -> LaunchConfig:
        return dill.loads(serialized_config) 

class Status(Enum):
    RUNNING = 1
    DONE = 2
    FAILED = 3

class AgentStatus:

    def __init__(self: AgentStatus, result: RunProcsResult, dummy = False):

        if dummy:
            self.status = Status.DONE
            self.failures = None    
            return

        self.failures = None
        if result is None:
            self.status = Status.RUNNING
        elif result.is_failed():
            self.status = Status.FAILED
            self.failures = result.failures
        else:
            self.status = Status.DONE

    def is_failed(self):
        return self.status == Status.FAILED
    
    def is_done(self):
        return self.status == Status.DONE
    
    def __repr__(self):
        return str(self.__dict__)


def launch(
    num_nodes: int = 4,
    num_processes: int = 4, # per node
    log_file: str = 'parallel_processing.log', # TODO: use
    ips_port_users: List[Tuple[str, int, str]] = [],
    backend : str = None, # TODO: check valid option passed
    func: Callable = None,
    **kwargs
):
    
    if not dist.is_available():
        raise RuntimeError("The torch.distributed package is not available.")

    # populate kwargs of target function early
    func = partial(func, **kwargs)
    #serialized_function = dill.dumps(func)

    # determine IP and an open port to run agent-launcher group from
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    launcher_port = get_open_port()

    # set some environmental variables. TODO: none of these env vars needed?
    os.environ["WORLD_SIZE"] = str(num_nodes * num_processes)
    os.environ["NODE_RANK"] = "0"
    os.environ["NPROC"] = str(num_processes)
    #os.environ["MASTER_ADDR"] = master_ip
    #os.environ["MASTER_PORT"] = str(master_port)

    # start agents on each node
    for i, (ip_forgn, port_forgn, user) in enumerate(ips_port_users):
        # connect via SSH
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ip_forgn, port_forgn, user) 
        # execute agent & disconnect
        # uses environment that multinode_spawner was executed in
        client.exec_command(f"{sys.executable} -u -m torchrunx {num_nodes+1} {i+1} {ip_address} {launcher_port} > /dev/null 2>&1 &")
        client.close()

    # create TCPStore for group initialization.
    launcher_store = dist.TCPStore(hostname, launcher_port, is_master=True)

    # initialize agent-launcher process group
    dist.init_process_group(backend="gloo", world_size=num_nodes+1, rank=0, store=launcher_store)
    # populate and broadcast agent parameters
    config = LaunchConfig(func, num_nodes, num_processes, backend)
    params = [config.serialize()]
    dist.broadcast_object_list(params)
    # participate in synchronization between agents, which is irrelevant to the launcher
    dist.broadcast_object_list([None, None], src=1)
    dummy_launch_status = AgentStatus(None, True)
    while True:
        # keep checking all agents...
        statuses: list[AgentStatus] = [None] * (num_nodes + 1)
        dist.all_gather_object(statuses, dummy_launch_status)

        # if any workers on any agent have failed
        if any(map(lambda s: s.is_failed(), statuses)):
            # terminate - the agents should also be exiting
            e = ""
            for i, s in enumerate(filter(lambda s: s.is_failed(), statuses)):
                for k, v in s.failures.items():
                    e += f"Node {i}, local worker {k} exited with error: {v.message['message']}\n"
                    e += f"{v.message['extraInfo']['py_callstack']}\n\n"
            raise RuntimeError(e)
        
        # else, check if everything's done
        if all(map(lambda s: s.is_done(), statuses)):
            # we can exit loop and gather return values
            break

    # wait for return values
    output = [None for i in range(num_nodes+1)]
    dist.gather_object({}, output, dst=0)
    
    # gather return values in {worker_rank: worker_return_value} format, and return
    result = {}
    for d in output:
        result.update(d)
    return result
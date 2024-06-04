from __future__ import annotations

import sys, getpass, time
import socket
from functools import partial
from typing import Callable
from enum import Enum
from datetime import timedelta

from torchrunx.utils import get_open_port

import dill
import paramiko

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.api import RunProcsResult

class LaunchConfig:

    def __init__(self: LaunchConfig, fn: Callable, world_size: int, node_worker_ranks: list[list[int]], backend: str) -> None:
        self.serialized_fn = dill.dumps(fn)
        self.world_size = world_size
        self.node_worker_ranks = node_worker_ranks
        self.backend = backend

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
    func: Callable,
    node_ips: list[str],
    num_workers: int = 4, # per node
    log_file: str = 'parallel_processing.log', # TODO: use
    user = getpass.getuser(),
    ssh_port = 22,
    backend : str = None,
    workers_per_node: list[int] = [], # overrides num_workers
    **kwargs
):
    
    if not dist.is_available():
        raise RuntimeError("The torch.distributed package is not available.")
    
    if backend not in ["gloo", "nccl", "gloo|nccl", None]:
        raise ValueError(f"backend must be one of 'gloo', 'nccl', 'gloo|nccl', or None (default, automatically determined), but '{backend}' was provided")
    
    num_nodes = len(node_ips)

    if workers_per_node != [] and len(workers_per_node) != num_nodes:
        raise ValueError(f"Number of nodes must match between node_ips and workers_per_node. Got {len(node_ips)=} and {len(workers_per_node)=}.")

    node_worker_ranks: list[list[int]] = []
    c = 0
    for n in range(num_nodes):
        node_workers = num_workers if workers_per_node == [] else workers_per_node[n]
        node_worker_ranks.append(list(range(c, c+node_workers)))
        c += node_workers

    world_size = num_nodes * num_workers if workers_per_node == [] else sum(workers_per_node)

    # populate kwargs of target function early
    func = partial(func, **kwargs)
    #serialized_function = dill.dumps(func)

    # determine IP and an open port to run agent-launcher group from
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    launcher_port = get_open_port()

    # set some environmental variables. TODO: none of these env vars needed?
    #os.environ["WORLD_SIZE"] = str(num_nodes * num_workers)
    #os.environ["NODE_RANK"] = "0"
    #os.environ["NPROC"] = str(num_workers)
    #os.environ["MASTER_ADDR"] = master_ip
    #os.environ["MASTER_PORT"] = str(master_port)

    # start agents on each node
    for i, ip_forgn in enumerate(node_ips):
        # connect via SSH
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ip_forgn, ssh_port, user) 
        # execute agent & disconnect
        # uses environment that multinode_spawner was executed in
        client.exec_command(f"{sys.executable} -u -m torchrunx {num_nodes+1} {i+1} {ip_address} {launcher_port} > /dev/null 2>&1 &")
        client.close()

    # create TCPStore for group initialization.
    launcher_store = dist.TCPStore(hostname, launcher_port, is_master=True)
    # initialize agent-launcher process group
    dist.init_process_group(backend="gloo", world_size=num_nodes+1, rank=0, store=launcher_store, timeout=timedelta(seconds=30))
    # populate and broadcast agent parameters
    config = LaunchConfig(func, world_size, node_worker_ranks, backend)
    params = [config]
    dist.broadcast_object_list(params)
    # participate in synchronization between agents, which is irrelevant to the launcher
    dist.broadcast_object_list([None, None], src=1)
    # gather pids of agents, in case they need to be manually terminated
    _pids = [None] * (num_nodes + 1)
    dist.gather_object(None, _pids)
    agent_pids = _pids[1:]
    # start monitoring loop
    dummy_launch_status = AgentStatus(None, True)
    while True:
        # keep checking all agents...
        statuses: list[AgentStatus] = [None] * (num_nodes + 1)
        try:
            dist.all_gather_object(statuses, dummy_launch_status)
        except:
            # kill all agents (most should be dead but some could be hanging)
            kill_agents(agent_pids, node_ips, ssh_port, user)
            # TODO: can we extract more info for this error?
            raise RuntimeError("One or more agents encountered an error.")

        # if any workers on any agent have failed
        if any(map(lambda s: s.is_failed(), statuses)):
            # terminate - the agents should also be exiting
            e = ""
            for i, s in filter(lambda s: s[1].is_failed(), enumerate(statuses)):
                for k, v in s.failures.items():
                    e += f"Node {i-1}, local worker {k} exited with error: {v.message['message']}\n"
                    e += f"{v.message['extraInfo']['py_callstack']}\n\n"
            raise RuntimeError(e)
        
        # else, check if everything's done
        if all(map(lambda s: s.is_done(), statuses)):
            # we can exit loop and gather return values
            break

    # wait for return values
    output = [None for i in range(num_nodes+1)]
    try:
        dist.gather_object({}, output, dst=0)
    except:
        kill_agents(agent_pids, ip_forgn, ssh_port, user)
        # TODO: can we extract more info for this error?
        raise RuntimeError("One or more agents encountered an error.")
    
    # gather return values in {worker_rank: worker_return_value} format, and return
    result = {}
    for d in output:
        result.update(d)
    return result

def kill_agents(pids: list[int], ips: list[str], ssh_port: int, user: str) -> None:
    for pid, ip_forgn in zip(pids, ips):
        # connect via SSH
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ip_forgn, ssh_port, user) 
        # execute agent & disconnect
        # uses environment that multinode_spawner was executed in
        client.exec_command(f"kill {pid} > /dev/null 2>&1 &")
        client.close()
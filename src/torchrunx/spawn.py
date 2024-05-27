from __future__ import annotations

import os, sys
import socket
from functools import partial
from typing import Callable, List, Tuple

from torchrunx.utils import get_open_port

import dill
import paramiko

import torch.distributed as dist


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
    launcher_store = dist.TCPStore(hostname, launcher_port, world_size=num_nodes*num_processes, is_master=True)

    # initialize agent-launcher process group
    dist.init_process_group(backend="gloo", world_size=num_nodes+1, rank=0, store=launcher_store)
    
    # populate and broadcast agent parameters
    config = LaunchConfig(func, num_nodes, num_processes, backend)
    params = [config.serialize()]
    dist.broadcast_object_list(params)
    
    # participate in synchronization between agents, which is irrelevant to the launcher
    dist.broadcast_object_list([None, None], src=1)

    # wait for return values
    output = [None for i in range(num_nodes+1)]
    dist.gather_object({}, output, dst=0)
    
    # gather return values in {worker_rank: worker_return_value} format, and return
    result = {}
    for d in output:
        result.update(d)
    # TODO: handle errors in agents, for now:
    assert result != {}, "All workers failed to execute"
    return result
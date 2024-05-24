import os, sys
import socket
from contextlib import closing
from functools import partial
from typing import Callable, List, Tuple

import dill
import paramiko

import torch.distributed as dist


def launch(
    num_nodes: int = 4,
    num_processes: int = 4, # per node
    timeout: int = 300, # TODO: unused
    max_retries: int = 3, # TODO: unused
    master_ip : str = '127.0.0.1', # TODO: should this be an argument? unused
    master_port_range : Tuple[int, int] = (20, 1024), # TODO: unused
    log_file: str = 'parallel_processing.log', # TODO: unused
    ips_port_users: List[Tuple[str, int, str]] = [], # TODO: unused
    messaging : str = "gloo", # TODO: unused
    func: Callable = None,
    **kwargs
):
    
    if not dist.is_available():
        raise RuntimeError("The torch.distributed package is not available.")

    # populate kwargs of target function early
    func = partial(func, **kwargs)
    serialized_function = dill.dumps(func)

    # determine IP and an open port to run agent-master group from
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        master_port = s.getsockname()[1]

    # set some environmental variables. TODO: only WORLD_SIZE, MASTER_PORT/ADDR required?
    os.environ["WORLD_SIZE"] = str(num_nodes * num_processes)
    os.environ["NODE_RANK"] = "0"
    os.environ["NPROC"] = str(num_processes)
    os.environ["MASTER_ADDR"] = master_ip
    os.environ["MASTER_PORT"] = str(master_port)

    # start agents on each node
    for i, (ip_forgn, port_forgn, user) in enumerate(ips_port_users):
        # connect via SSH
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(ip_forgn, port_forgn, user) 
        # execute agent & disconnect
        # uses environment that multinode_spawner was executed in
        client.exec_command(f"{sys.executable} -u -m torchrunx {num_nodes+1} {i+1} {ip_address} {master_port} > /dev/null 2>&1 &")
        client.close()

    # initialize agnet-master process group
    dist.init_process_group(backend="gloo", world_size=num_nodes+1, rank=0)
    
    # populate and broadcast agent parameters
    params = [{'func': serialized_function, 'args': dill.dumps(tuple()), 
               'nodes': num_nodes, 'nprocs': num_processes}]
    dist.broadcast_object_list(params)
    
    # participate in synchronization between agents, which is irrelevant to the master
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
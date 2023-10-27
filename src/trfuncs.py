from functools import partial

import os
import socket
import random
from typing import List, Tuple

from multiprocessing import Process, Queue

import dill
import paramiko


def my_function(x, y):
    return x + y

serialized_function = dill.dumps(my_function)

def spawnx(config, func, **kwargs):
    
    os.environ["WORLD_SIZE"] = str(config.num_nodes * config.num_procs)
    os.environ["NODE_RANK"] = "0"
    os.environ["NPROC"] = str(config.num_procs)
    os.environ["MASTER_ADDR"] = config.master_ip
    
    func_to_serialize = partial(func, **kwargs)

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    
    print(f"Hostname: {hostname}")
    print(f"IP Address: {ip_address}") 

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        master_port = s.getsockname()[1]
    
    os.environ["MASTER_PORT"] = master_port
    
    for i, (ip_forgn, port_forgn, user) in enumerate(config.ips_port_users):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
        # so do not need to manually add
        
        client.connect(ip_forgn, port_forgn, user) 
        # potentially need either pub key or user name
        
        
        # stdin, stdout, stderr = client.exec_command(f'echo "{serialized_function}" > my_function.pkl')
        stdin, stdout, stderr = client.exec_command(
            f'python -m torchrunx "{serialized_function}" {config.num_nodes} {config.num_processes} {ip_address} {master_port} {i}'
            )
        print(stdout.read().decode())
        client.close()
    


def torchrunx(serialized_func, num_nodes, num_processes, ip_master, port_master, rank):
    os.environ["WORLD_SIZE"] = str(num_nodes * num_processes)
    os.environ["NODE_RANK"] = str(rank)
    os.environ["NPROC"] = str(num_processes)
    os.environ["MASTER_ADDR"] = ip_master
    os.environ["MASTER_PORT"] = str(port_master)
    
    deserialized_func = dill.loads(serialized_function)
    
    args = {i: tuple() for i in range(num_processes)}
    envs = {i: {} for i in range(num_procs)}

    if log_dir is None:
        log_dir = tempfile.mkdtemp()
        
    ctx: MultiprocessContext = start_processes(
        name="distributed_function",
        entrypoint=entrypoint,
        args={i: (i, fn, *args[i]) for i in args},
        envs=envs,
        log_dir=log_dir,
        start_method="spawn",
    )
    result: RunProcsResult = ctx.wait()
    return result.return_values

def entrypoint(index: int, fn: Callable, *args):
    os.environ["LOCAL_RANK"] = str(index)
    os.environ["RANK"] = str(index)
    dist.init_process_group(backend="nccl", world_size=int(os.environ["WORLD_SIZE"]), rank=index)
    return fn(*args)
    
# no gloo, nccl in python, just mpi4py 

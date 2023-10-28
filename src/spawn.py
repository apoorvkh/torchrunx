import os
import socket
from contextlib import closing
from functools import partial
from typing import Callable, List, Tuple

import dill
import paramiko


def multinode_spawner(
    num_nodes: int = 4,
    num_processes: int = 4,  # per node
    timeout: int = 300,    
    max_retries: int = 3,
    master_ip : str = '127.0.0.1',
    master_port_range : Tuple[int, int] = (20, 1024),
    log_file: str = 'parallel_processing.log',
    ips_port_users: List[Tuple[str, int, str]] = [],
    messaging : str = "gloo",
    func: Callable = None,
    **kwargs
):
    os.environ["WORLD_SIZE"] = str(num_nodes * num_processes)
    os.environ["NODE_RANK"] = "0"
    os.environ["NPROC"] = str(num_processes)
    os.environ["MASTER_ADDR"] = master_ip

    func = partial(func, **kwargs)
    serialized_function = dill.dumps(func)

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    
    print(f"Hostname: {hostname}")
    print(f"IP Address: {ip_address}") 

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        master_port = s.getsockname()[1]
    
    os.environ["MASTER_PORT"] = master_port
    
    for i, (ip_forgn, port_forgn, user) in enumerate(ips_port_users):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
        # so do not need to manually add
        
        client.connect(ip_forgn, port_forgn, user) 
        # potentially need either pub key or user name

        # stdin, stdout, stderr = client.exec_command(f'echo "{serialized_function}" > my_function.pkl')
        # stdin, stdout, stderr = client.exec_command(
        #     f'python -m torchrunx "{serialized_function}" {num_nodes} {num_processes} {ip_address} {master_port} {i}'
        # )
        stdin, stdout, stderr = client.exec_command(
            f'python -m torchrunx'
        )
        print(stdout.read().decode())
        client.close()

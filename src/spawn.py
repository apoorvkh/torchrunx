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

def print_env():
    for key in sorted(os.environ.keys()):
        if not (
            key.startswith(("SLURM_", "SUBMITIT_"))
            or key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        ):
            continue
        value = os.environ[key]
        print(f"{key}={value}")


class Task:
    def __call__(self):
        # print_env()
        print("exporting PyTorch distributed environment variables")
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")
        # print_env()

        # Using the (default) env:// initialization method
        torch.distributed.init_process_group(backend="nccl")
        assert dist_env.rank == torch.distributed.get_rank()
        assert dist_env.world_size == torch.distributed.get_world_size()

        # Actual task / computation
        tensor = dist_env.rank * torch.ones(1).cuda()

        time.sleep(120)

        torch.distributed.all_reduce(tensor)
        if dist_env.rank == 0:
            result = list(tensor)
            print(result)
            return result

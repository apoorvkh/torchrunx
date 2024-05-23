import os, sys, socket
import tempfile
from typing import Callable
from contextlib import closing
import time

import dill
import torch.distributed as dist
# import tyro # what does this do
from torch.distributed.elastic.multiprocessing import start_processes
from torch.distributed.elastic.multiprocessing.api import MultiprocessContext, RunProcsResult

import torchrunx.entry as entry

def main(world_size, rank):
    # TODO initialize communication with controller
    # torch.distributed.init_process_group(gloo), ranks >= 1
    #world_size = int(os.environ['SLURM_JOB_NUM_NODES'])*int(os.environ['SLURM_NPROCS'])
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
    params = [None]
    dist.broadcast_object_list(params)
    params = params[0]
    # TODO: get serialized function and arguments
    serialized_function: str = params['func']
    #fn = dill.loads(serialized_function)

    # TODO: get node, process, rank information
    num_nodes: int =  params['nodes']#1
    num_processes: int = params['nprocs'] # int(os.environ['SLURM_NPROCS']) #

    if rank == 1: 
        master_hostname = socket.gethostname()
        master_ip = socket.gethostbyname(master_hostname)
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            master_port = s.getsockname()[1]
        master = [master_ip, master_port]
    else:
        master = [None, None]
    dist.broadcast_object_list(master, src=1)

    master_ip: str = master[0]
    master_port: int = master[1]
    #rank: int = params[''] #0

    #os.environ["WORLD_SIZE"] = str(num_nodes * num_processes)
    #print(f"world size: {num_nodes * num_processes}")
    #os.environ["NODE_RANK"] = str(rank-1)
    #os.environ["NPROC"] = str(num_processes)
    #os.environ["MASTER_ADDR"] = master_ip
    #os.environ["MASTER_PORT"] = str(master_port)

    args = {i: dill.loads(params['args']) for i in range(num_processes)}
    envs = {i: {"RANK": (rank-1)*num_processes + i, 
                "LOCAL_RANK": i, 
                "WORLD_SIZE": str(num_nodes * num_processes),
                "MASTER_ADDR": master_ip,
                "MASTER_PORT": str(master_port)} 
            for i in range(num_processes)} # "MASTER_ADDR": master_ip, "MASTER_PORT": master_port

    log_dir = None
    if log_dir is None:
        log_dir = tempfile.mkdtemp() #  f"/users/pcurtin1/torchrunx/log/{rank}/" # 
    
    #print("Spawning...")
    ctx: MultiprocessContext = start_processes(
        name="distributed_function",
        entrypoint=entry.entrypoint,
        args={i: (serialized_function, *args[i]) for i in args},
        envs=envs,
        log_dir=log_dir,
        start_method="spawn",
    )
    #print("Waiting...")
    result: RunProcsResult = ctx.wait()
    #print("All workers terminated. Sending return values to master.")
    if result.failures:
        print(result.failures)
    # need to modify the keys in result.return_values to reflect global rank not local
    return_values = {k + (rank-1)*num_processes: v for k, v in result.return_values.items()}
    dist.gather_object(return_values, dst=0)
    #dist.all_gather_object([None for i in range(world_size)], result.return_values)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = sys.argv[3]
    os.environ["MASTER_PORT"] = sys.argv[4]
    main(int(sys.argv[1]), int(sys.argv[2]))

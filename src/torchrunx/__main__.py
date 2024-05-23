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
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)

    # receieve parameters from master
    _params = [None]
    dist.broadcast_object_list(_params)
    params = params[0]

    serialized_function: str = params['func']
    num_nodes: int =  params['nodes']
    num_processes: int = params['nprocs']
    arguments = dill.loads(params['args'])

    # broadcast/receive master worker's IP and port
    if rank == 1: 
        # rank 1 agent is responsible for rank 0 worker, aka the "master worker"
        # thus grab a port on this agent's node
        master_hostname = socket.gethostname()
        master_ip = socket.gethostbyname(master_hostname)
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            master_port = s.getsockname()[1]
        master = [master_ip, master_port]
    else:
        # else, we listen for the broadcast
        master = [None, None]

    dist.broadcast_object_list(master, src=1)

    master_ip: str = master[0]
    master_port: int = master[1]

    # set arguments and environmental variables for each worker
    args = {i: arguments for i in range(num_processes)}
    envs = {i: {"RANK": str((rank-1)*num_processes + i), 
                "LOCAL_RANK": str(i), 
                "WORLD_SIZE": str(num_nodes * num_processes),
                "MASTER_ADDR": master_ip,
                "MASTER_PORT": str(master_port)} for i in range(num_processes)}
    
    # logging directory
    log_dir = None
    if log_dir is None:
        log_dir = tempfile.mkdtemp() #  f"/users/pcurtin1/torchrunx/log/{rank}/" # 
    
    # spawn workers
    ctx: MultiprocessContext = start_processes(
        name="distributed_function",
        entrypoint=entry.entrypoint,
        args={i: (serialized_function, *args[i]) for i in args},
        envs=envs,
        log_dir=log_dir,
        start_method="spawn",
    )
    
    # wait for all terminated
    result: RunProcsResult = ctx.wait()
    
    # handle errors, TODO: determine what to do here, e.g. throw error?
    if result.failures:
        print(result.failures)

    # gather return values, and send them to master
    # need to modify the keys in result.return_values to reflect global ranks not local ranks or workers
    return_values = {k + (rank-1)*num_processes: v for k, v in result.return_values.items()}
    dist.gather_object(return_values, dst=0)

if __name__ == "__main__":
    # parse arguments, TODO: use argparse
    # TODO: WORLD_SIZE and RANK variables could be set rather than main having arguments...
    os.environ["MASTER_ADDR"] = sys.argv[3]
    os.environ["MASTER_PORT"] = sys.argv[4]
    main(int(sys.argv[1]), int(sys.argv[2]))

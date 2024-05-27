import sys, socket
import tempfile
import torch.distributed as dist
# import tyro # what does this do
from torch.distributed.elastic.multiprocessing import start_processes
from torch.distributed.elastic.multiprocessing.api import MultiprocessContext, RunProcsResult

import torchrunx.entry as entry

from torchrunx.utils import get_open_port
from torchrunx.spawn import LaunchConfig

def main(world_size, rank, launcher_ip, launcher_port):

    # create client TCPStore for initializing launcher-agent process group
    store = dist.TCPStore(launcher_ip, launcher_port, world_size=world_size)

    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank, store=store)

    # receieve parameters from launcher
    _params = [None]
    dist.broadcast_object_list(_params)
    config = LaunchConfig.deserialize(_params[0])

    serialized_function: str = config.serialized_fn
    num_nodes: int =  config.num_nodes
    num_processes: int = config.num_processes
    backend: str = config.backend
    #arguments = dill.loads(params['args'])

    # broadcast/receive launcher worker's IP and port
    if rank == 1: 
        # rank 1 agent is responsible for rank 0 worker, aka the "master worker"
        # thus grab a port on this agent's node
        master_hostname = socket.gethostname()

        master_ip = socket.gethostbyname(master_hostname)
        master_port = get_open_port()
        master = [master_ip, master_port]
    else:
        # else, we listen for the broadcast
        master = [None, None]

    dist.broadcast_object_list(master, src=1)

    master_ip: str = master[0]
    master_port: int = master[1]

    # set arguments and environmental variables for each worker
    #args = {i: arguments for i in range(num_processes)}
    envs = {i: {"RANK": str((rank-1)*num_processes + i), 
                "LOCAL_RANK": str(i), 
                "WORLD_SIZE": str(num_nodes * num_processes)
                } for i in range(num_processes)}
    
    # logging directory
    log_dir = None
    if log_dir is None:
        log_dir = tempfile.mkdtemp() #  f"/users/pcurtin1/torchrunx/log/{rank}/" # 
    
    # spawn workers
    ctx: MultiprocessContext = start_processes(
        name="distributed_function",
        entrypoint=entry.entrypoint,
        args={i: (serialized_function, master_ip, master_port, backend) for i in range(num_processes)}, # backend=None for now
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
    # need to modify the keys in result.return_values to reflect global ranks not local ranks of workers
    return_values = {k + (rank-1)*num_processes: v for k, v in result.return_values.items()}
    dist.gather_object(return_values, dst=0)

if __name__ == "__main__":
    # parse arguments, TODO: use argparse
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))

import sys, socket
import tempfile
import torch.distributed as dist
# import tyro # what does this do
from torch.distributed.elastic.multiprocessing import start_processes, DefaultLogsSpecs
from torch.distributed.elastic.multiprocessing.api import MultiprocessContext
from datetime import timedelta

import torchrunx.entry as entry

from torchrunx.utils import get_open_port
from torchrunx.spawn import LaunchConfig, AgentStatus

def main(world_size: int, rank: int, launcher_ip: str, launcher_port: int):

    # create client TCPStore for initializing launcher-agent process group
    store = dist.TCPStore(launcher_ip, launcher_port)
    #print("got store, trying setup")
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank, store=store, timeout=timedelta(seconds=30))

    # receieve parameters from launcher
    _params = [None]
    dist.broadcast_object_list(_params)
    config: LaunchConfig = _params[0]

    serialized_function = config.serialized_fn
    worker_world_size = config.world_size
    #num_nodes =  config.num_nodes
    worker_ranks = config.node_worker_ranks[rank-1]
    num_workers = len(worker_ranks)
    backend = config.backend
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
    envs = {i: {"RANK": str(worker_ranks[i]), 
                "LOCAL_RANK": str(i), 
                "WORLD_SIZE": str(worker_world_size)
                } for i in range(num_workers)}
    
    # logging directory
    log_dir = None
    if log_dir is None:
        log_dir = tempfile.mkdtemp() #  f"/users/pcurtin1/torchrunx/log/{rank}/" # 
    
    # spawn workers
    ctx: MultiprocessContext = start_processes(
        name="distributed_function",
        entrypoint=entry.entrypoint,
        args={i: (serialized_function, master_ip, master_port, backend) for i in range(num_workers)},
        envs=envs,
        logs_specs=DefaultLogsSpecs(log_dir=log_dir),
        start_method="spawn",
    )
    done = False
    while True:

        # determine status of this agent, five-second timeout
        if not done:
            result = ctx.wait(5)
        status = AgentStatus(result)
        done = status.is_done()
        # grab statuses of other agents
        statuses: list[AgentStatus] = [None] * world_size
        try:
            dist.all_gather_object(statuses, status)
        except:
            ctx.close()
            return
        # if any workers on any agent have failed
        if any(map(lambda s: s.is_failed(), statuses)):
            # terminate local workers and exit
            ctx.close()
            return
        
        # else, check if everything's done
        if all(map(lambda s: s.is_done(), statuses)):
            # we can exit loop and gather return values
            break

        # otherwise, continue...

    return_values = {worker_ranks[k]: v for k, v in result.return_values.items()}
    dist.gather_object(return_values, dst=0)

if __name__ == "__main__":
    # parse arguments, TODO: use argparse
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))

import dill, torch, os
import torch.distributed as dist

def entrypoint(fn: bytes, master_ip: str, master_port: int, backend: str, *args):
    _fn = dill.loads(fn)

    # Initialize TCPStore for group
    is_master = os.environ["RANK"] == "0"
    world_size = int(os.environ["WORLD_SIZE"])
    store = dist.TCPStore(master_ip, master_port, world_size=world_size, is_master=is_master)

    if backend is None:
        backend = "gloo|nccl" if torch.cuda.is_available() else "gloo"
    rank = int(os.environ["RANK"])
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank, store=store)
    return _fn(*args)
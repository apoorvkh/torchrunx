import dill, torch
import torch.distributed as dist

def entrypoint(fn: bytes, *args):
    _fn = dill.loads(fn)
    # I believe these do not need to be passed, due to them being environmental vars already:
    # initializes both gloo and nccl if possible.
    dist.init_process_group(backend="gloo|nccl" if torch.cuda.is_available() else "gloo")
    return _fn(*args)
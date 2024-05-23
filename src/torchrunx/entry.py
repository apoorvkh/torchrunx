def entrypoint(fn: bytes, *args):
    # idk if these imports need to be in here or outside of here. Outside means that when importing 
    # torchrunx.entry in torchrunx.__main__ the imports will re-run which should be fine actually
    import os, dill
    import torch.distributed as dist

    _fn = dill.loads(fn)

    # I believe these do not need to be passed, due to them being environmental vars already:
    #world_size=int(os.environ["WORLD_SIZE"]), rank=int(os.environ["RANK"])
    dist.init_process_group(backend="gloo")     
    return _fn(*args)
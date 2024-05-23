def entrypoint(fn: bytes, *args): #rank: int, local_rank: int,
    # idk if these imports need to be in here or outside of here. Outside means that when importing 
    # torchrunx.entry in torchrunx.__main__ the imports will re-run which should be fine actually
    import os, dill
    import torch.distributed as dist

    _fn = dill.loads(fn)
    #os.environ["LOCAL_RANK"] #= str(local_rank)
    #os.environ["RANK"] #= str(rank)
    for var in ["RANK", 
                "LOCAL_RANK", 
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT"]:
        print(os.environ[var])
    #print(f"{rank}, {local_rank}, {_fn}, {args}, {os.environ['WORLD_SIZE']}")
    dist.init_process_group(backend="gloo", world_size=int(os.environ["WORLD_SIZE"]), rank=os.environ["RANK"])
    return _fn(*args)
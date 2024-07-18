import os

import torch
import torch.distributed as dist

import torchrunx


def test_launch():
    result = torchrunx.launch(
        func=simple_matmul,
        func_kwargs={},
        hostnames=torchrunx.slurm_hosts(),
        workers_per_host=torchrunx.slurm_workers(),
    )

    t = True
    for i in range(len(result)):
        t = t and torch.all(result[i] == result[0])

    assert t, "Not all tensors equal"

    dist.destroy_process_group()


def simple_matmul():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if rank == 0:
        w = torch.rand((100, 100), device=device)  # in_dim, out_dim
    else:
        w = torch.zeros((100, 100), device=device)

    dist.broadcast(w, 0)

    i = torch.rand((500, 100), device=device)  # batch, dim
    o = torch.matmul(i, w)

    dist.all_reduce(o, op=dist.ReduceOp.SUM)
    print(i)
    return o.detach().cpu()


if __name__ == "__main__":
    test_launch()

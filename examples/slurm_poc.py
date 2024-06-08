import os

import torch
import torch.distributed as dist

from torchrunx import launch

# this is not a pytest test, but a functional test designed to be run on a slurm allocation


def test_launch():
    result = launch(
        func=simple_matmul,
        func_kwargs={},
        use_slurm=True,
    )

    for i in range(len(result)):
        assert torch.all(result[i] == result[0]), "Not all tensors equal"
    print(result[0])
    print("PASS")


def simple_matmul():
    rank = int(os.environ["RANK"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if rank == 0:
        w = torch.rand((100, 100), device=device)  # in_dim, out_dim
    else:
        w = torch.zeros((100, 100), device=device)

    dist.broadcast(w, 0)

    i = torch.rand((500, 100), device=device)  # batch, dim
    o = torch.matmul(i, w)

    dist.all_reduce(o, op=dist.ReduceOp.SUM)
    return o.detach().cpu()


if __name__ == "__main__":
    test_launch()

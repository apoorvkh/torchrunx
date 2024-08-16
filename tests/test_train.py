import os
import sys

sys.path.append("../src")

import torch.distributed as dist

import torchrunx


def worker():
    import torch

    class TwoLinLayerNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(10, 10, bias=False)
            self.b = torch.nn.Linear(10, 1, bias=False)

        def forward(self, x):
            a = self.a(x)
            b = self.b(x)
            return (a, b)

    local_rank = int(os.environ["LOCAL_RANK"])
    print("init model")
    model = TwoLinLayerNet().to(local_rank)
    print("init ddp")
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    inp = torch.randn(10, 10).to(local_rank)
    print("train")

    for _ in range(20):
        output = ddp_model(inp)
        loss = output[0] + output[1]
        loss.sum().backward()


def test_distributed_train():
    torchrunx.launch(
        worker,
        hostnames=torchrunx.slurm_hosts(),
        workers_per_host=torchrunx.slurm_workers(),
        backend="nccl",
    )


if __name__ == "__main__":
    test_distributed_train()

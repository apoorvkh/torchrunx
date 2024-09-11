import os

import torchrunx as trx


def worker() -> None:
    import torch

    class TwoLayerNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Linear(10, 10, bias=False)
            self.b = torch.nn.Linear(10, 1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a = self.a(x)
            b = self.b(a)
            return b

    local_rank = int(os.environ["LOCAL_RANK"])
    print("init model")
    model = TwoLayerNN().to(local_rank)
    print("init ddp")
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    inp = torch.randn(10, 10).to(local_rank)
    print("train")

    for _ in range(20):
        output = ddp_model(inp)
        loss = output.sum()
        loss.backward()


def test_distributed_train() -> None:
    trx.launch(
        worker,
        hostnames="slurm",
        workers_per_host="slurm",
        backend="nccl",
    )


if __name__ == "__main__":
    test_distributed_train()

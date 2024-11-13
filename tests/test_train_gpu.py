import os

import torch

import torchrunx as trx


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(10, 10, bias=False)
        self.b = torch.nn.Linear(10, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


def worker() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    print("init model")
    model = MLP().to(local_rank)
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
        backend="nccl",
    )


if __name__ == "__main__":
    test_distributed_train()

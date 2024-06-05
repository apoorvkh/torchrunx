import os
import subprocess
import socket
from torchrunx import launch


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

    torch.cuda.set_device(0)
    print("init model")
    model = TwoLinLayerNet().cuda()
    print("init ddp")
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])

    inp = torch.randn(10, 10).cuda()
    print("train")

    for _ in range(20):
        output = ddp_model(inp)
        loss = output[0] + output[1]
        loss.sum().backward()


def resolve_node_ips(nodelist):
    # Expand the nodelist into individual hostnames
    hostnames = (
        subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
        .decode()
        .strip()
        .split("\n")
    )
    # Resolve each hostname to an IP address
    ips = [socket.gethostbyname(hostname) for hostname in hostnames]
    return ips


if __name__ == "__main__":
    launch(
        worker,
        resolve_node_ips(os.environ["SLURM_JOB_NODELIST"]),
        num_workers=int(os.environ["SLURM_NTASKS_PER_NODE"]),
        backend="nccl",
    )

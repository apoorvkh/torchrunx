import os
import socket
import subprocess
import torch
from torchrunx import launch
# this is not a pytest test, but a functional test designed to be run on a slurm allocation


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


def get_ips_port_users(nodelist, port, user):
    ips = resolve_node_ips(nodelist)
    # Pair each IP with the specified port and user
    ips_port_users = [(ip, port, user) for ip in ips]
    return ips_port_users


def slurm_ips_port_users():
    nodelist = os.environ["SLURM_JOB_NODELIST"]
    port = 22  # SSH port
    user = "pcurtin1"  # Replace with the appropriate username
    return get_ips_port_users(nodelist, port, user)


def test_launch():
    result = launch(
        func=simple_matmul,
        func_kwargs={},
        hostnames=resolve_node_ips(os.environ["SLURM_JOB_NODELIST"]),
        workers_per_host=int(os.environ["SLURM_NTASKS_PER_NODE"]),
    )

    for i in range(len(result)):
        assert torch.all(result[i] == result[0]), "Not all tensors equal"
    print(result[0])
    print("PASS")


def simple_matmul():
    import torch
    import os
    import torch.distributed as dist

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

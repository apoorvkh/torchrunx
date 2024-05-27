import os, socket, subprocess, torch
from torchrunx import launch
# this is not a pytest test, but a functional test designed to be run on a slurm allocation

def resolve_node_ips(nodelist):
    # Expand the nodelist into individual hostnames
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', nodelist]).decode().strip().split('\n')
    # Resolve each hostname to an IP address
    ips = [socket.gethostbyname(hostname) for hostname in hostnames]
    return ips

def get_ips_port_users(nodelist, port, user):
    ips = resolve_node_ips(nodelist)
    # Pair each IP with the specified port and user
    ips_port_users = [(ip, port, user) for ip in ips]
    return ips_port_users

def slurm_ips_port_users():
    nodelist = os.environ['SLURM_JOB_NODELIST']
    port = 22  # SSH port
    user = 'pcurtin1'  # Replace with the appropriate username
    return get_ips_port_users(nodelist, port, user)

def test_launch():
    #print(datetime.datetime.now())
    # Here we use Slurm environment variables directly
    world_size = int(os.environ['SLURM_NTASKS'])
    #node_rank = os.environ['SLURM_NODEID']
    nproc = int(os.environ['SLURM_CPUS_ON_NODE'])
    #master_addr = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    #master_port = os.environ.get('MASTER_PORT', '29500')  # Default if not set
    #print(world_size, nproc)

    result = launch(
        num_nodes=int(world_size),
        num_processes=nproc,
        ips_port_users=slurm_ips_port_users(),
        func=simple_matmul
    )

    assert result != {}, "Computation failed"
    for i in range(len(result)):
        assert torch.all(result[i] == result[0]), "Not all tensors equal"
    print(result[0])
    print("PASS")
    

def simple_matmul():
    import torch, os
    import torch.distributed as dist
    """ Run collective communication. """
    #group = dist.new_group([0, 1, 2, 3])
    rank = int(os.environ["RANK"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if rank == 0:
        w = torch.rand((100, 100), device=device) # in_dim, out_dim
    else:
        w = torch.zeros((100, 100), device=device)

    dist.broadcast(w, 0)

    i = torch.rand((500, 100), device=device) # batch, dim

    o = torch.matmul(i, w)

    dist.all_reduce(o, op=dist.ReduceOp.SUM)
    # note: return must be a cpu tensor...
    return o.detach().cpu()

if __name__ == "__main__":
    test_launch()

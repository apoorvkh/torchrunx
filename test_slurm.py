import socket, torch
from torchrunx.spawn import multinode_spawner
import os 
import socket
import os
import subprocess

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

def test_multinode_spawner():
    # Here we use Slurm environment variables directly
    world_size = os.environ['SLURM_NTASKS']
    node_rank = os.environ['SLURM_NODEID']
    nproc = os.environ['SLURM_CPUS_ON_NODE']
    master_addr = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    master_port = os.environ.get('MASTER_PORT', '29500')  # Default if not set
    #print(world_size, node_rank, nproc, master_addr, master_port)

    result = multinode_spawner(
        num_nodes=int(world_size),
        num_processes=2,
        ips_port_users=slurm_ips_port_users(),
        func=run_all_reduce,
    )

    for i in range(len(result)):
        assert torch.all(result[i] == result[0]), "Not all tensors equal"
    print(result)
    print("PASS")
    

def run_all_reduce():
    import torch
    import torch.distributed as dist
    """ Run collective communication. """
    #group = dist.new_group([0, 1, 2, 3])
    tensor = torch.rand(100)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

if __name__ == "__main__":
    test_multinode_spawner()

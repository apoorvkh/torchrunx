import socket, torch
from torchrunx.spawn import multinode_spawner
import os 
import socket
import os
import subprocess
import funcs

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

# @pytest.fixture
# def mock_socket():
#     mock = Mock()
#     mock.bind = Mock()
#     mock.getsockname = MagicMock(return_value=('', 12345))
#     return mock

# @pytest.fixture
# def mock_paramiko_client():
#     client = Mock()
#     client.connect = Mock()
#     readmock = Mock()
#     readmock.read = Mock(return_value=b'success')
#     client.exec_command = Mock(return_value=(Mock(), readmock, Mock()))
#     client.close = Mock()
#     return client

# @patch('src.spawn.paramiko.SSHClient')
# @patch('src.spawn.socket.socket')
def test_multinode_spawner():
    # Here we use Slurm environment variables directly
    world_size = os.environ['SLURM_NTASKS']
    node_rank = os.environ['SLURM_NODEID']
    nproc = os.environ['SLURM_CPUS_ON_NODE']
    master_addr = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    master_port = os.environ.get('MASTER_PORT', '29500')  # Default if not set
    print(world_size, node_rank, nproc, master_addr, master_port)

    multinode_spawner(
        num_nodes=int(world_size),
        num_processes=2,
        ips_port_users=slurm_ips_port_users(),
        func=run_all_reduce,
    )

def run_all_reduce(rank, world_size):
    import torch
    import torch.distributed as dist
    """ Run collective communication. """
    group = dist.new_group([0, 1, 2, 3])
    tensor = torch.rand(100)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor

if __name__ == "__main__":
    test_multinode_spawner()

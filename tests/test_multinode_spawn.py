import pytest
import socket
from unittest.mock import Mock, patch, MagicMock
from src.torchrunx.spawn import multinode_spawner
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

@pytest.fixture
def slurm_ips_port_users():
    nodelist = os.environ['SLURM_JOB_NODELIST']
    port = 22  # SSH port
    user = 'pcurtin1'  # Replace with the appropriate username
    return get_ips_port_users(nodelist, port, user)

@pytest.fixture
def mock_socket():
    mock = Mock()
    mock.bind = Mock()
    mock.getsockname = MagicMock(return_value=('', 12345))
    return mock

@pytest.fixture
def mock_paramiko_client():
    client = Mock()
    client.connect = Mock()
    readmock = Mock()
    readmock.read = Mock(return_value=b'success')
    client.exec_command = Mock(return_value=(Mock(), readmock, Mock()))
    client.close = Mock()
    return client

@patch('src.spawn.paramiko.SSHClient')
@patch('src.spawn.socket.socket')
def test_multinode_spawner(mock_socket_class, mock_ssh_client, mock_socket, mock_paramiko_client, slurm_ips_port_users):
    mock_socket_class.return_value = mock_socket
    mock_ssh_client.return_value = mock_paramiko_client
    # Here we use Slurm environment variables directly
    world_size = os.environ['SLURM_NTASKS']
    node_rank = os.environ['SLURM_NODEID']
    nproc = os.environ['SLURM_CPUS_ON_NODE']
    master_addr = os.environ['SLURM_LAUNCH_NODE_IPADDR']
    master_port = os.environ.get('MASTER_PORT', '29500')  # Default if not set

    multinode_spawner(
        num_nodes=int(world_size),
        num_processes=1,
        ips_port_users=slurm_ips_port_users,
        func=lambda x: x * 2 
    )

    mock_socket_class.assert_called_with(socket.AF_INET, socket.SOCK_STREAM)
    mock_socket.bind.assert_called_with(("", 0))

    assert mock_paramiko_client.connect.call_count == int(world_size)
    assert mock_paramiko_client.exec_command.call_count == int(world_size)
    assert mock_paramiko_client.close.call_count == int(world_size)

    # Asserting that the environment variables are set as expected
    assert os.environ['WORLD_SIZE'] == world_size
    # assert os.environ['NODE_RANK'] == node_rank # not sure about this...
    assert os.environ['NPROC'] == nproc

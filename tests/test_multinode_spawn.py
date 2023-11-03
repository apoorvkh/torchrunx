import pytest
import socket
from unittest.mock import Mock, patch, MagicMock
from src.spawn import multinode_spawner  


@pytest.fixture
def mock_socket():
    mock = Mock()
    mock.bind = Mock()
    mock.getsockname = Mock(return_value=('', 12345))
    return mock

@pytest.fixture
def mock_paramiko_client():
    client = Mock()
    client.connect = Mock()
    client.exec_command = Mock(return_value=(Mock(), Mock(return_value=b'success'), Mock()))
    client.close = Mock()
    return client


@patch('your_module.socket.socket', return_value=mock_socket())
@patch('your_module.paramiko.SSHClient', return_value=mock_paramiko_client())
def test_multinode_spawner(mock_socket_class, mock_ssh_client):
    with patch.dict('os.environ', {'WORLD_SIZE': '', 'NODE_RANK': '', 'NPROC': '', 'MASTER_ADDR': '', 'MASTER_PORT': ''}):
        multinode_spawner(
            num_nodes=2,
            ips_port_users=[('192.168.1.1', 22, 'user1'), ('192.168.1.2', 22, 'user2')],
            func=lambda x: x * 2 
        )

        mock_socket_class.assert_called_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_socket().bind.assert_called_with(("", 0))

        assert mock_ssh_client().connect.call_count == 2
        assert mock_ssh_client().exec_command.call_count == 2
        assert mock_ssh_client().close.call_count == 2

        assert os.environ['WORLD_SIZE'] == '8'
        assert os.environ['NODE_RANK'] == '0'
        assert os.environ['NPROC'] == '4'
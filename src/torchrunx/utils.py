import socket
import fabric
from contextlib import closing


def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


def ssh_exec(command, ip, ssh_port, user):
    c = fabric.Connection(host=ip, user=user, port=ssh_port)
    c.run(command)
    c.close()

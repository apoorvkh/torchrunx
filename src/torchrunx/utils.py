import socket, paramiko
from contextlib import closing

def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port

def ssh_exec(command, ip, ssh_port, user):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # connect via SSH
    client.connect(ip, ssh_port, user)
    # execute command
    client.exec_command(command)
    # close connection
    client.close()
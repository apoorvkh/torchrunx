import socket
import fabric
import os
from contextlib import closing
from dotenv import dotenv_values

def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


def ssh_exec(command, ip, ssh_port, user):
    c = fabric.Connection(host=ip, user=user, port=ssh_port)
    c.run(command)
    c.close()

def get_env(env_filepath: str = None) -> "dict[str, str]":
    explicit = ["LIBRARY_PATH", "LD_LIBRARY_PATH", "PATH"]

    env = {k: v for k, v in os.environ.items() if k in explicit}

    prefixes = ["CUDA", "NCCL", "OMP", "PYTHON", "HF", "TORCH", "TRITON"]

    for k, v in os.environ.items():
        if any([k.startswith(prefix) for prefix in prefixes]):
            env[k] = v

    if env_filepath is not None:
        env.update(dotenv_values(env_filepath))

    return env
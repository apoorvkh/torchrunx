from __future__ import annotations
from typing import TypeVar, Any
from typing_extensions import Self
import os

import socket
import fabric
from contextlib import closing
import cloudpickle
import torch.distributed as dist
import torch


class Serializable:
    @property
    def serialized(self) -> bytes:
        return cloudpickle.dumps(self)

    @classmethod
    def from_serialized(cls, serialized: bytes) -> Self:
        return cloudpickle.loads(serialized)


def get_open_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    return port


def execute_ssh_command(
    command: str, hostname: str, ssh_config_file: str | os.PathLike | None = None
) -> None:
    with fabric.Connection(
        host=hostname, config=fabric.Config(runtime_ssh_path=ssh_config_file)
    ) as conn:
        conn.run(f"{command} > /dev/null 2>&1 &")


def serialize(object: Any) -> bytes:
    return cloudpickle.dumps(object)


def deserialize(serialized: bytes) -> Any:
    return cloudpickle.loads(serialized)


T = TypeVar("T")


def broadcast(
    object: Any,
    src: int = 0,
    group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
):
    data = [serialize(object)]
    dist.broadcast_object_list(object_list=data, src=src, group=group, device=device)
    return deserialize(data[0])


def gather(
    object: Any, dst: int = 0, group: dist.ProcessGroup | None = None
) -> list | None:
    object_bytes = serialize(object)

    object_gather_list = None
    if dist.get_rank(group) == dst:
        object_gather_list = [None] * dist.get_world_size(group)

    dist.gather_object(
        obj=object_bytes, object_gather_list=object_gather_list, dst=dst, group=group
    )

    if object_gather_list is not None:
        object_gather_list = [deserialize(o) for o in object_gather_list]

    return object_gather_list


def all_gather(object: Any, group: dist.ProcessGroup | None = None) -> list:
    object_bytes = serialize(object)
    object_list = [bytes()] * dist.get_world_size(group)
    dist.all_gather_object(object_list=object_list, obj=object_bytes, group=group)
    object_list = [deserialize(o) for o in object_list]
    return object_list

import os
import tempfile

import pytest
import torch
import torch.distributed as dist

import torchrunx as trx
from torchrunx.logging_utils import default_handlers


def test_simple_localhost():
    def dist_func():
        rank = int(os.environ["RANK"])

        if rank == 0:
            w = torch.rand((100, 100))  # in_dim, out_dim
        else:
            w = torch.zeros((100, 100))

        dist.broadcast(w, 0)

        i = torch.rand((500, 100))  # batch, dim
        o = torch.matmul(i, w)

        dist.all_reduce(o, op=dist.ReduceOp.SUM)

        print(i)

        return o.detach()

    r = trx.launch(
        func=dist_func,
        func_kwargs={},
        workers_per_host=2,
        backend="gloo",  # log_dir="./test_logs"
    )

    assert torch.all(r[0] == r[1])


def test_logging():
    def dist_func():
        rank = int(os.environ["RANK"])
        print(f"worker rank: {rank}")

    tmp = tempfile.mkdtemp()
    trx.launch(
        func=dist_func,
        func_kwargs={},
        workers_per_host=2,
        backend="gloo",
        log_handlers=default_handlers(hostnames=["localhost"], workers_per_host=[2], log_dir=tmp),
    )

    log_files = next(os.walk(tmp), (None, None, []))[2]

    assert len(log_files) == 3

    for file in log_files:
        with open(f"{tmp}/{file}", "r") as f:
            contents = f.read()
            if file.endswith("[0].log"):
                assert "worker rank: 0\n" in contents
            elif file.endswith("[1].log"):
                assert "worker rank: 1\n" in contents
            else:
                assert "starting processes" in contents


def test_error():
    def error_func():
        raise ValueError("abcdefg")

    with pytest.raises(RuntimeError) as excinfo:
        trx.launch(
            func=error_func,
            func_kwargs={},
            workers_per_host=1,
            backend="gloo",
            # log_dir=tempfile.mkdtemp(),
        )

    assert "abcdefg" in str(excinfo.value)

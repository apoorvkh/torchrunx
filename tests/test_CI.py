import os
import tempfile

import pytest
import torch
import torch.distributed as dist

import torchrunx as trx


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

    tmp = tempfile.mkdtemp()
    os.environ["TORCHRUNX_DIR"] = tmp

    r = trx.launch(
        func=dist_func,
        func_kwargs={},
        workers_per_host=2,
        backend="gloo",  # log_dir="./test_logs"
    )

    results = next(iter(r.values()))
    assert torch.all(results[0] == results[1])


def test_logging():
    def dist_func():
        rank = int(os.environ["RANK"])
        print(f"worker rank: {rank}")

    tmp = tempfile.mkdtemp()
    os.environ["TORCHRUNX_LOG_DIR"] = tmp

    trx.launch(
        func=dist_func,
        func_kwargs={},
        workers_per_host=2,
        backend="gloo",
    )

    log_files = next(os.walk(tmp), (None, None, []))[2]

    assert len(log_files) == 3

    for file in log_files:
        with open(f"{tmp}/{file}") as f:
            contents = f.read()
            print(contents)
            if file.endswith("[0].log"):
                assert "worker rank: 0\n" in contents
            elif file.endswith("[1].log"):
                assert "worker rank: 1\n" in contents
            else:
                assert "starting processes" in contents


def test_error():
    def error_func():
        raise ValueError("abcdefg")

    tmp = tempfile.mkdtemp()
    os.environ["TORCHRUNX_DIR"] = tmp

    with pytest.raises(ValueError) as excinfo:
        trx.launch(
            func=error_func,
            func_kwargs={},
            workers_per_host=1,
            backend="gloo",
        )

    assert "abcdefg" in str(excinfo.value)


if __name__ == "__main__":
    test_simple_localhost()

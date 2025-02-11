import datetime
import os
import tempfile
import time
from pathlib import Path
from typing import NoReturn

import pytest
import torch
import torch.distributed as dist

import torchrunx as trx


def test_simple_localhost() -> None:
    def dist_func() -> torch.Tensor:
        rank = int(os.environ["RANK"])

        w = torch.rand((100, 100)) if rank == 0 else torch.zeros((100, 100))

        dist.broadcast(w, 0)

        i = torch.rand((500, 100))  # batch, dim
        o = torch.matmul(i, w)

        dist.all_reduce(o, op=dist.ReduceOp.SUM)

        print(i)

        return o.detach()

    tmp = tempfile.mkdtemp()
    os.environ["TORCHRUNX_DIR"] = tmp

    r = trx.launch(
        dist_func,
        workers_per_host=2,
        backend="gloo",  # log_dir="./test_logs"
    )

    assert torch.all(r.rank(0) == r.rank(1))


def test_logging() -> None:
    def dist_func() -> None:
        rank = int(os.environ["RANK"])
        print(f"worker rank: {rank}")

    tmp = tempfile.mkdtemp()
    os.environ["TORCHRUNX_LOG_DIR"] = tmp

    num_workers = 2

    before_timestamp = datetime.datetime.now()

    time.sleep(1)

    trx.launch(
        dist_func,
        workers_per_host=num_workers,
        backend="gloo",
    )

    after_timestamp = datetime.datetime.now()

    log_dirs = next(os.walk(tmp), (None, [], None))[1]

    assert len(log_dirs) == 1

    # this should error if mis-formatted
    log_timestamp = datetime.datetime.fromisoformat(log_dirs[0])

    assert before_timestamp <= log_timestamp <= after_timestamp

    log_files = next(os.walk(f"{tmp}/{log_dirs[0]}"), (None, None, []))[2]

    assert len(log_files) == num_workers + 1

    for file in log_files:
        with Path(f"{tmp}/{log_dirs[0]}/{file}").open() as f:
            contents = f.read()
            print(contents)
            if file.endswith("[0].log"):
                assert "worker rank: 0\n" in contents
            elif file.endswith("[1].log"):
                assert "worker rank: 1\n" in contents


def test_error() -> None:
    def error_func() -> NoReturn:
        msg = "abcdefg"
        raise ValueError(msg)

    tmp = tempfile.mkdtemp()
    os.environ["TORCHRUNX_DIR"] = tmp

    with pytest.raises(ValueError) as excinfo:  # noqa: PT011
        trx.launch(
            error_func,
            workers_per_host=1,
            backend="gloo",
        )

    assert "abcdefg" in str(excinfo.value)


if __name__ == "__main__":
    test_simple_localhost()

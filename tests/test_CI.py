import os
import sys
import tempfile
import time
from threading import Thread

import pytest
import torch
import torch.distributed as dist

from torchrunx.utils import AgentPayload

sys.path.append("../src")

import torchrunx  # noqa: I001


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

    r = torchrunx.launch(
        func=dist_func, func_kwargs={}, workers_per_host=2, backend="gloo", log_dir="./test_logs"
    )

    assert torch.all(r[0] == r[1])


def test_logging():
    def dist_func():
        rank = int(os.environ["RANK"])
        print(f"worker rank: {rank}")

    tmp = tempfile.mkdtemp()
    torchrunx.launch(
        func=dist_func, func_kwargs={}, workers_per_host=2, backend="gloo", log_dir=tmp
    )

    log_files = next(os.walk(tmp), (None, None, []))[2]

    assert len(log_files) == 3

    for file in log_files:
        with open(f"{tmp}/{file}", "r") as f:
            if file.endswith("0.log"):
                assert f.read() == "worker rank: 0\n"
            elif file.endswith("1.log"):
                assert f.read() == "worker rank: 1\n"
            else:
                contents = f.read()
                assert "worker rank: 0" in contents
                assert "worker rank: 1" not in contents


def test_error():
    def error_func():
        raise ValueError("abcdefg")

    with pytest.raises(RuntimeError) as excinfo:
        torchrunx.launch(
            func=error_func,
            func_kwargs={},
            workers_per_host=1,
            backend="gloo",
            log_dir="./test_logs",
        )

    assert "abcdefg" in str(excinfo.value)


def test_timeout():
    def dist_func():
        time.sleep(30)

    pids = []

    original = torchrunx.launcher.LauncherAgentGroup.sync_payloads

    def wrap(self, payload):
        r = original(self, payload)
        _r: list[AgentPayload] = r[1:]  # pyright: ignore[reportAssignmentType]
        pids.extend([p.process_id for p in _r])
        return r

    torchrunx.launcher.LauncherAgentGroup.sync_payloads = wrap

    def suspend():
        while len(pids) == 0:
            time.sleep(0.5)
        print(pids[0])
        os.system(f"kill -TSTP {pids[0]}")

    thr = Thread(target=suspend)
    with pytest.raises(RuntimeError) as excinfo:
        thr.start()
        torchrunx.launch(
            func=dist_func,
            func_kwargs={},
            workers_per_host=1,
            backend="gloo",
            log_dir="./test_logs",
        )
    thr.join()
    os.system(f"kill {pids[0]}")
    assert "Timed out" in str(excinfo.value)

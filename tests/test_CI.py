import os
import shutil
import sys

import torch
import torch.distributed as dist

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
        func=dist_func,
        func_kwargs={},
        workers_per_host=2,
        backend="gloo",
    )

    assert torch.all(r[0] == r[1])

    dist.destroy_process_group()


def test_logging():
    def dist_func():
        rank = int(os.environ["RANK"])
        print(f"worker rank: {rank}")

    try:
        shutil.rmtree("./test_logs", ignore_errors=True)
    except FileNotFoundError:
        pass

    torchrunx.launch(
        func=dist_func, func_kwargs={}, workers_per_host=2, backend="gloo", log_dir="./test_logs"
    )

    log_files = next(os.walk("./test_logs"), (None, None, []))[2]

    assert len(log_files) == 3

    for file in log_files:
        with open("./test_logs/" + file, "r") as f:
            if file.endswith("0.log"):
                assert f.read() == "worker rank: 0\n"
            elif file.endswith("1.log"):
                assert f.read() == "worker rank: 1\n"
            else:
                contents = f.read()
                assert "worker rank: 0" in contents
                assert "worker rank: 1" in contents

    # clean up
    shutil.rmtree("./test_logs", ignore_errors=True)

    dist.destroy_process_group()

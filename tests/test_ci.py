import datetime
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import NoReturn
from unittest import mock

import pytest
import torch
import torch.distributed as dist

import torchrunx as trx
from torchrunx.utils.environment import execute_command


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

    r = trx.Launcher(
        workers_per_host=2,
        backend="gloo",
    ).run(dist_func)

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

    trx.Launcher(
        workers_per_host=num_workers,
        backend="gloo",
    ).run(
        dist_func,
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
        trx.Launcher(
            workers_per_host=1,
            backend="gloo",
        ).run(
            error_func,
        )

    assert "abcdefg" in str(excinfo.value)


def test_execute_command_inherits_output_when_not_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = mock.Mock()
    popen = mock.Mock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", popen)

    result = execute_command("echo output", "localhost")

    assert result == ("", "")
    assert popen.call_args.kwargs["stdout"] is None
    assert popen.call_args.kwargs["stderr"] is None
    process.communicate.assert_not_called()


def test_execute_command_captures_requested_output(monkeypatch: pytest.MonkeyPatch) -> None:
    process = mock.Mock()
    process.communicate.return_value = ("stdout", "stderr")
    popen = mock.Mock(return_value=process)
    monkeypatch.setattr(subprocess, "Popen", popen)

    result = execute_command("echo output", "localhost", return_stdout_stderr=True)

    assert result == ("stdout", "stderr")
    assert popen.call_args.kwargs["stdout"] == subprocess.PIPE
    assert popen.call_args.kwargs["stderr"] == subprocess.PIPE
    process.communicate.assert_called_once_with()


if __name__ == "__main__":
    test_simple_localhost()

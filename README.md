# torchrunx 🔥

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchrunx)](https://github.com/apoorvkh/torchrunx/blob/main/pyproject.toml)
[![PyTorch Version](https://img.shields.io/badge/torch-%3E%3D2.0-orange)](https://github.com/pytorch/pytorch)
[![PyPI - Version](https://img.shields.io/pypi/v/torchrunx)](https://pypi.org/project/torchrunx/)
![Tests](https://img.shields.io/github/actions/workflow/status/apoorvkh/torchrunx/.github%2Fworkflows%2Fmain.yml)
[![Docs](https://readthedocs.org/projects/torchrunx/badge/?version=stable)](https://torchrunx.readthedocs.io)
[![GitHub License](https://img.shields.io/github/license/apoorvkh/torchrunx)](https://github.com/apoorvkh/torchrunx/blob/main/LICENSE)

By [Apoorv Khandelwal](http://apoorvkh.com) and [Peter Curtin](https://github.com/pmcurtin)

**The easiest way to run PyTorch on multiple GPUs or machines.**

---

**`torchrunx`** is a more convenient, *functional* replacement for CLI-based distributed PyTorch launchers (`torchrun`, `accelerate launch`, `deepspeed`, etc).

Simply put, you can distribute PyTorch functions from Python like:

```python
def train(num_steps: int): ... # implemented below

import torchrunx as trx

# Run train(num_steps=10) on 2 machines with 2 GPUs each

result = trx.launch(
    func=train,
    func_kwargs=dict(num_steps=10),
    hostnames=["localhost", "other_node"],
    workers_per_host=2
)

trained_model = result.rank(0)
torch.save(trained_model.state_dict(), "model.pth")
```

<details>
  <summary>Training function (expand)</summary>

```python
import os
import torch

def train(num_steps: int = 5):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    model = torch.nn.Linear(10, 10).to(local_rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters())

    for step in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(5, 10))
        labels = torch.randn(5, 10).to(local_rank)
        torch.nn.functional.mse_loss(outputs, labels).backward()
        optimizer.step()

    if rank == 0:
        return model
```
</details>

**Refer to our [API](https://torchrunx.readthedocs.io/stable/api.html) and [Advanced Usage Guide](https://torchrunx.readthedocs.io/stable/advanced.html) for many more capabilities!**

## Installation

```bash
pip install torchrunx
```

**Requires:** Linux (+ SSH & shared filesystems if using multiple machines)

## Why?

This library uniquely offers:

1. **An automatic launcher that just works for everyone** 🚀

No system-specific dependencies and orchestration for *automatic* multi-node distribution. `torchrunx` is an SSH-based, pure-Python library that is universally easy to install.

2. **Conventional CLI commands** 🖥️

Run familiar commands, like `python my_script.py ...`, and customize arguments as you wish.

In contrast to launchers that override the `python` executable in a cumbersome way (e.g. `torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=100.43.331.111 --master_port=1234 my_script.py ...`).

3. **Support for more complex workflows in a single script** 🎛️

Your workflow may have independent steps that need different parallelizations (e.g. training on 8 GPUs, testing on 1 GPU; comparing throughput on 4, then 8 GPUs; and so forth). CLI-based launchers naively parallelize the entire script for exactly *N* GPUs. In contrast, our library treats these steps in a modular way and permits *degrees* of parallelism in a single script.

We clean memory leaks (which are unfortunately common in PyTorch) as we go, so previous steps won't crash or adversely affect future steps.

4. **Better handling of system failures. No more zombies!** 🧟

With `torchrun`, your "work" is inherently coupled to your main Python process. If the system kills one of your workers (e.g. due to RAM OOM or segmentation faults), there is no way to fail gracefully in Python. Your processes might hang for at least 10 minutes (the NCCL timeout) or become perpetual zombies.

`torchrunx` decouples "launcher" and "worker" processes. If the system kills a worker, our launcher immediately raises a `WorkerFailure` exception, which users can handle as they wish. We always clean up all nodes, so no more zombies!

5. **Bonus features** 🎁

- Fine-grained, custom handling of logging, environment variables, and exception propagation. We have nice defaults too: no more interleaved logs and irrelevant exceptions!
- No need to manually set up a [`dist.init_process_group`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
- We automatically detect and infer settings from SLURM environments.
- Start multi-node training from Python notebooks!

On our [roadmap](https://github.com/apoorvkh/torchrunx/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement): higher-order parallelism, support for debuggers, fuller typing, and more!

## Examples with other libraries

<details>
  <summary>Accelerate</summary>

  ```python
  ```
</details>

<details>
  <summary>HF Trainer</summary>

  ```python
  ```
</details>

<details>
  <summary>Deepspeed</summary>

  ```python
  ```
</details>

<details>
  <summary>PyTorch Lightning</summary>

  ```python
  ```
</details>

<details>
  <summary>MosaicML Composer</summary>

  ```python
  ```
</details>

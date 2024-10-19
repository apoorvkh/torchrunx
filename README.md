# torchrunx 🔥

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchrunx)](https://github.com/apoorvkh/torchrunx/blob/main/pyproject.toml)
[![PyTorch Version](https://img.shields.io/badge/torch-%3E%3D2.0-orange)](https://github.com/pytorch/pytorch)
[![PyPI - Version](https://img.shields.io/pypi/v/torchrunx)](https://pypi.org/project/torchrunx/)
![Tests](https://img.shields.io/github/actions/workflow/status/apoorvkh/torchrunx/.github%2Fworkflows%2Fmain.yml)
[![Docs](https://readthedocs.org/projects/torchrunx/badge/?version=stable)](https://torchrunx.readthedocs.io)
[![GitHub License](https://img.shields.io/github/license/apoorvkh/torchrunx)](https://github.com/apoorvkh/torchrunx/blob/main/LICENSE)

By [Apoorv Khandelwal](http://apoorvkh.com) and [Peter Curtin](https://github.com/pmcurtin)

**Automatically distribute PyTorch functions onto multiple machines or GPUs**

## Installation

```bash
pip install torchrunx
```

**Requires:** Linux. Shared filesystem & SSH access if using multiple machines.

## Why should I use this?

Whether you have 1 GPU, 8 GPUs, or 8 machines:

Convenience:

- If you don't want to set up [`dist.init_process_group`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) yourself
- If you don't want to manually SSH into every machine (and `torchrun --master-ip --master-port ...` and babysit hanging failures)

Robustness:

- If you want to run a complex, _modular_ workflow in one script
  - no worries about memory leaks or OS failures
  - don't parallelize your entire script: just the functions you want

Features:

- Our launch utility is super _Pythonic_
    - Return objects from your distributed functions
    - Run `python script.py` instead of `torchrun script.py`
    - Launch functions from Python Notebooks
- Fine-grained control over logging, environment variables, exception handling, etc.
- Automatic integration with SLURM

## Minimal example

Here's a simple example where we "train" a model on two nodes (with 2 GPUs each). You can also use `transformers.Trainer` (or similar) which handles all the multi-GPU (DDP) code for you.

```python
import os
import torch

def train():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    model = torch.nn.Linear(10, 10).to(local_rank)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(ddp_model.parameters())
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(5, 10))
    labels = torch.randn(5, 10).to(local_rank)
    torch.nn.functional.mse_loss(outputs, labels).backward()
    optimizer.step()

    if rank == 0:
        return model
```

```python
import torchrunx as trx

if __name__ == "__main__":
    trained_model = trx.launch(
        func=train,
        hostnames=["localhost", "other_node"],
        workers_per_host=2
    ).value(rank=0)

    torch.save(trained_model.state_dict(), "model.pth")
```

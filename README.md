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

## Minimal example

Here's a simple example where we distribute `train_model` to two hosts (with 2 GPUs each):

```python
def train_model(model, train_dataset):
    trained_model = train(model, train_dataset)

    if int(os.environ["RANK"]) == 0:
        torch.save(learned_model, 'model.pt')
        return 'model.pt'

    return None
```

```python
import torchrunx as trx

learned_model_path = trx.launch(
    func=train_model,
    func_kwargs={'model': my_model, 'train_dataset': mnist_train},
    hostnames=["localhost", "other_node"],
    workers_per_host=2
).value(0)  # return from rank 0 (first worker on "localhost")
```

## Why should I use this?

[`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) is a hammer. `torchrunx` is a chisel.

Whether you have 1 GPU, 8 GPUs, or 8 machines:

Convenience:

- If you don't want to set up [`dist.init_process_group`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) yourself
- If you want to run `python myscript.py` instead of `torchrun myscript.py`
- If you don't want to manually SSH and run `torchrun --master-ip --master-port ...` on every machine (and if you don't want to babysit these machines for hanging failures)

Robustness:

- If you want to run a complex, _modular_ workflow in one script
  - no worries about memory leaks or OS failures
  - don't parallelize your entire script: just the functions you want

Features:

- Our launch utility is super _Pythonic_
- If you want to run distributed PyTorch functions from Python Notebooks.
- Automatic integration with SLURM

Why not?

- We don't support fault tolerance via torch elastic. Probably only useful if you are using 1000 GPUs. Maybe someone can make a PR.

## More complicated example

We could also launch multiple functions, on different nodes:

```python
def train_model(model, train_dataset):
    trained_model = train(model, train_dataset)

    if int(os.environ["RANK"]) == 0:
        torch.save(learned_model, 'model.pt')
        return 'model.pt'

    return None

def test_model(model_path, test_dataset):
    model = torch.load(model_path)
    accuracy = inference(model, test_dataset)
    return accuracy
```

```python
import torchrunx as trx

learned_model_path = trx.launch(
    func=train_model,
    func_kwargs={'model': my_model, 'train_dataset': mnist_train},
    hostnames=["beefy-node"],
    workers_per_host=2
).value(0)  # return from rank 0 (first worker on "beefy-node")



accuracy = trx.launch(
    func=test_model,
    func_kwargs={'model_path': learned_model_path, 'test_dataset': mnist_test},
    hostnames=["localhost"],
    workers_per_host=1
).value(0)

print(f'Accuracy: {accuracy}')
```

# torchrunx 🔥

By [Apoorv Khandelwal](http://apoorvkh.com) and [Peter Curtin](https://github.com/pmcurtin)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchrunx)](https://github.com/apoorvkh/torchrunx/blob/main/pyproject.toml)
[![PyPI - Version](https://img.shields.io/pypi/v/torchrunx)](https://pypi.org/project/torchrunx/)
![Tests](https://img.shields.io/github/actions/workflow/status/apoorvkh/torchrunx/.github%2Fworkflows%2Fmain.yml)
[![Docs](https://readthedocs.org/projects/torchrunx/badge/?version=stable)](https://torchrunx.readthedocs.io)
[![GitHub License](https://img.shields.io/github/license/apoorvkh/torchrunx)](https://github.com/apoorvkh/torchrunx/blob/main/LICENSE)

Automatically launch PyTorch functions onto multiple machines or GPUs

## Installation

```bash
pip install torchrunx
```

Requirements:
- Operating System: Linux
- Python >= 3.8.1
- PyTorch >= 2.0
- Shared filesystem & SSH between hosts

## Features

- Distribute PyTorch functions to multiple GPUs or machines
- `torchrun` with the convenience of a Python function
- Integration with SLURM

Advantages:

- Self-cleaning: avoid memory leaks!
- Better for complex workflows
- Doesn't parallelize the whole script: just what you want
- Run distributed functions from Python Notebooks

## Usage

Here's a simple example where we distribute `distributed_function` to two hosts (with 2 GPUs each):

```python
def train_model(model, dataset):
    trained_model = train(model, dataset)

    if int(os.environ["RANK"]) == 0:
        torch.save(learned_model, 'model.pt')
        return 'model.pt'

    return None
```

```python
import torchrunx as trx

model_path = trx.launch(
    func=train_model,
    func_kwargs={'model': my_model, 'training_dataset': mnist_train},
    hostnames=["localhost", "other_node"],
    workers_per_host=2
)["localhost"][0]  # return from rank 0 (first worker on "localhost")
```

We could also launch multiple functions, with different GPUs:

```python
def train_model(model, dataset):
    trained_model = train(model, dataset)

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

model_path = trx.launch(
    func=train_model,
    func_kwargs={'model': my_model, 'training_dataset': mnist_train},
    hostnames=["localhost", "other_node"],
    workers_per_host=2
)["localhost"][0]  # return from rank 0 (first worker on "localhost")



accuracy = trx.launch(
    func=test_model,
    func_kwargs={'model': learned_model, 'test_dataset': mnist_test},
    hostnames=["localhost"],
    workers_per_host=1
)["localhost"][0]

print(f'Accuracy: {accuracy}')
```
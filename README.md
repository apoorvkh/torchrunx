# torchrunx 🔥

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchrunx)](https://github.com/apoorvkh/torchrunx/blob/main/pyproject.toml)
[![PyPI - Version](https://img.shields.io/pypi/v/torchrunx)](https://pypi.org/project/torchrunx/)
[![Docs](https://readthedocs.org/projects/torchrunx/badge/?version=latest)](https://torchrunx.readthedocs.io)
[![GitHub License](https://img.shields.io/github/license/apoorvkh/torchrunx)](https://github.com/apoorvkh/torchrunx/blob/main/LICENSE)

Automatically launch functions and initialize distributed PyTorch environments on multiple machines

## Installation

```bash
pip install torchrunx
```

Requirements:
- Operating System: Linux
- Python >= 3.8.1
- PyTorch >= 2.0
- Shared filesystem & passwordless SSH between hosts

## Usage

```python
# Simple example
def distributed_function():
    pass
```

```python
import torchrunx as trx

trx.launch(
    func=distributed_function,
    func_kwargs={},
    hostnames=["node1", "node2"],  # or just: ["localhost"]
    workers_per_host=2
)
```

### In a SLURM allocation

```python
trx.launch(
    # ...
    hostnames=trx.slurm_hosts(),
    workers_per_host=trx.slurm_workers()
)
```

## Compared to other tools

## Contributing

We use the [`pixi`](https://pixi.sh) package manager. Simply [install `pixi`](https://pixi.sh/latest/#installation) and run `pixi shell` in this repository. We use `ruff` for linting and formatting, `pyright` for static type checking, and `pytest` for testing. We build for `PyPI` and `conda-forge`. Our release pipeline is powered by Github Actions.

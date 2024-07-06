# torchrunx 🔥

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchrunx)
![PyPI - Version](https://img.shields.io/pypi/v/torchrunx)
![Docs](https://readthedocs.org/projects/torchrunx/badge/?version=latest)
![GitHub License](https://img.shields.io/github/license/apoorvkh/torchrunx)

Automatically launch functions and initialize distributed PyTorch environments on multiple machines

## Installation

```bash
pip install torchrunx
```

Requirements:
- Python >= 3.8.1
- PyTorch >= 2.0
- Shared filesystem & passwordless SSH between hosts

## Usage

Example distributed function:
```python
```

```python
import torchrunx

torchrunx.launch(
    func=distributed_function,
    func_kwargs={},
    hostnames=["localhost"],
    workers_per_host=2
)
```

### Multiple machines

```python
torchrunx.launch(
    # ...
    hostnames=["node1", "node2"],
    workers_per_host=2
)
```

### In a SLURM allocation

```python
torchrunx.launch(
    # ...
    hostnames=torchrunx.slurm_hosts(),
    workers_per_host=torchrunx.slurm_workers()
)
```

## Compared to other tools

## Contributing

We use the [`pixi`](https://pixi.sh) package manager. Simply [install `pixi`](latest/#installation) and run `pixi shell` in this repository. We use `ruff` for linting and formatting, `pyright` for static type checking, and `pytest` for testing. We build for `PyPI` and `conda-forge`. Our release pipeline is powered by Github Actions.

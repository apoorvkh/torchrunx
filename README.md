# torchrunx

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchrunx)
![PyPI - Version](https://img.shields.io/pypi/v/torchrunx)
![GitHub License](https://img.shields.io/github/license/apoorvkh/torchrunx)

## Installation

```bash
pip install torchrunx
```

## Usage

```python

import torchrunx

torchrunx.launch(
    func=function,
    func_kwargs={},
    hostnames=["localhost"],
    workers_per_host=2
)
```

## Contributing

Development environment:
1. [Install pixi](https://pixi.sh/latest/#installation)
2. `pixi shell`

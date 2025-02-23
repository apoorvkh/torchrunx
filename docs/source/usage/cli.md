# From the CLI

## With `argparse`

We provide some utilities to extend an {obj}`argparse.ArgumentParser` with arguments for building a {obj}`torchrunx.Launcher`.

```python
from argparse import ArgumentParser
from torchrunx.integrations.parsing import add_torchrunx_argument_group, launcher_from_args

if __name__ == '__main__':
    parser = ArgumentParser()
    add_torchrunx_argument_group(parser)
    args = parser.parse_args()

    launcher = launcher_from_args(args)
    launcher.run(...)
```

`python ... --help` then results in:

```{eval-rst}
.. literalinclude:: ../artifacts/argparse_cli_help.txt
```

## With automatic CLI tools

We can also automatically populate {mod}`torchrunx.Launcher` arguments using most CLI tools, e.g. [`tyro`](https://brentyi.github.io/tyro/) or any that [generate interfaces from dataclasses](https://brentyi.github.io/tyro/goals_and_alternatives).

```python
import torchrunx
import tyro

if __name__ == "__main__":
    launcher = tyro.cli(torchrunx.Launcher)
    results = launcher.run(...)
```

`python ... --help` then results in:

```{eval-rst}
.. literalinclude:: ../artifacts/tyro_cli_help.txt
  :lines: 3-
```

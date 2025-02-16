# CLI Integration

We can automatically populate {mod}`torchrunx.Launcher` arguments using most CLI tools, e.g. [`tyro`](https://brentyi.github.io/tyro/) or any that [generate interfaces from dataclasses](https://brentyi.github.io/tyro/goals_and_alternatives).

```python
import torchrunx
import tyro

def distributed_function():
    ...

if __name__ == "__main__":
    launcher = tyro.cli(torchrunx.Launcher)
    launcher.run(distributed_function)
```

`python ... --help` then results in:

```{eval-rst}
.. literalinclude:: ../artifacts/cli_help.txt
  :lines: 3-
```

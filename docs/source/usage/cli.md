# From the CLI

We can automatically populate {mod}`torchrunx.Launcher` arguments using most CLI tools, e.g. [`tyro`](https://brentyi.github.io/tyro/) or any that [generate interfaces from dataclasses](https://brentyi.github.io/tyro/goals_and_alternatives).

```python
import torchrunx
import tyro

if __name__ == "__main__":
    launcher = tyro.cli(torchrunx.Launcher)
    results = launcher.run(...)
```

`python ... --help` then results in:

```{eval-rst}
.. literalinclude:: ../artifacts/cli_help.txt
  :lines: 3-
```

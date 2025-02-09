# CLI Integration

We can use {mod}`torchrunx.Launcher` to populate arguments from the CLI (e.g. with [tyro](https://brentyi.github.io/tyro/)):

```python
import torchrunx as trx
import tyro

def distributed_function():
    pass

if __name__ == "__main__":
    launcher = tyro.cli(trx.Launcher)
    launcher.run(distributed_function)
```

`python ... --help` then results in:

```bash
╭─ options ─────────────────────────────────────────────╮
│ -h, --help           show this help message and exit  │
│ --hostnames {[STR [STR ...]]}|{auto,slurm}            │
│                      (default: auto)                  │
│ --workers-per-host INT|{[INT [INT ...]]}|{auto,slurm} │
│                      (default: auto)                  │
│ --ssh-config-file {None}|STR|PATH                     │
│                      (default: None)                  │
│ --backend {None,nccl,gloo,mpi,ucc,auto}               │
│                      (default: auto)                  │
│ --timeout INT        (default: 600)                   │
│ --default-env-vars [STR [STR ...]]                    │
│                      (default: PATH LD_LIBRARY ...)   │
│ --extra-env-vars [STR [STR ...]]                      │
│                      (default: )                      │
│ --env-file {None}|STR|PATH                            │
│                      (default: None)                  │
╰───────────────────────────────────────────────────────╯
```

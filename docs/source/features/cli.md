# CLI Integration

We can automatically populate {mod}`torchrunx.Launcher` arguments using most CLI tools (those that generate interfaces from Data Classes, e.g. [tyro](https://brentyi.github.io/tyro/)):

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

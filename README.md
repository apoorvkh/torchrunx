# torchrunx 🔥

[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fapoorvkh%2Ftorchrunx%2Fmain%2Fpyproject.toml)](https://github.com/apoorvkh/torchrunx/blob/main/pyproject.toml)
[![PyTorch Version](https://img.shields.io/badge/torch-%3E%3D2.0-orange)](https://github.com/pytorch/pytorch)
[![PyPI - Version](https://img.shields.io/pypi/v/torchrunx)](https://pypi.org/project/torchrunx/)
[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://torchrun.xyz)
![Tests](https://img.shields.io/github/actions/workflow/status/apoorvkh/torchrunx/.github%2Fworkflows%2Fmain.yml)
[![GitHub License](https://img.shields.io/github/license/apoorvkh/torchrunx)](https://github.com/apoorvkh/torchrunx/blob/main/LICENSE)

By [Apoorv Khandelwal](https://apoorvkh.com) and [Peter Curtin](https://github.com/pmcurtin)

**The easiest way to run PyTorch on multiple GPUs or machines.**

---

**`torchrunx`** is a *functional* utility for distributing PyTorch code across devices. This is a [more convenient, robust, and featureful](#torchrunx-uniquely-offers) alternative to CLI-based launchers, like `torchrun`, `accelerate launch`, and `deepspeed`.

```bash
pip install torchrunx
```

Requires: Linux (+ SSH & shared filesystem if using multiple machines)

---

**Vanilla Example: Training a model on 2 machines with 2 GPUs each**

Dummy distributed training function:

```python
from __future__ import annotations
import os
import torch
import torch.nn as nn

def train(model: nn.Module, num_steps: int = 5) -> nn.Module | None:
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    model.to(local_rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters())

    for step in range(10):
        optimizer.zero_grad()

        inputs = torch.randn(5, 10).to(local_rank)
        labels = torch.randn(5, 10).to(local_rank)
        outputs = ddp_model(inputs)

        torch.nn.functional.mse_loss(outputs, labels).backward()
        optimizer.step()

    if rank == 0:
        return model.cpu()
```

Launching training with `torchrunx`:

```python
import torchrunx

results = torchrunx.Launcher(
    hostnames = ["localhost", "second_machine"],
    workers_per_host = 2
).run(
    train,
    model = nn.Linear(10, 10),
    num_steps = 10
)

trained_model: nn.Module = results.rank(0)
torch.save(trained_model.state_dict(), "output/model.pth")
```

**See examples where we fine-tune LLMs (e.g. GPT-2 on WikiText) using:**
  - [Transformers](https://torchrun.xyz/examples/transformers.html)
  - [DeepSpeed](https://torchrun.xyz/examples/deepspeed.html)
  - [PyTorch Lightning](https://torchrun.xyz/examples/lightning.html)
  - [Accelerate](https://torchrun.xyz/examples/accelerate.html)

**Refer to our [API](https://torchrun.xyz/api.html) and [Advanced Usage Guide](https://torchrun.xyz/advanced.html) for many more capabilities!**

---

## `torchrunx` uniquely offers

1. **An automatic launcher that "just works" for everyone** 🚀

> `torchrunx` is an SSH-based, pure-Python library that is universally easy to install.<br>
> No system-specific dependencies and orchestration for *automatic* multi-node distribution.

2. **Conventional CLI commands** 🖥️

> Run familiar commands, like `python my_script.py ...`, and customize arguments as you wish.
> 
> Other launchers override `python` in a cumbersome way: e.g. `torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=100.43.331.111 --master_port=1234 my_script.py ...`.

3. **Support for more complex workflows in a single script** 🎛️

> Your workflow may have independent steps that need different parallelizations (e.g. training on 8 GPUs, testing on 1 GPU; comparing throughput on 4, then 8 GPUs; and so forth). CLI-based launchers naively parallelize the entire script for exactly *N* GPUs. In contrast, our library treats these steps in a modular way and permits *degrees* of parallelism in a single script.
>
> 
> We clean memory leaks as we go, so previous steps won't crash or adversely affect future steps.

4. **Better handling of system failures. No more zombies!** 🧟

> With `torchrun`, your "work" is inherently coupled to your main Python process. If the system kills one of your workers (e.g. due to RAM OOM or segmentation faults), there is no way to fail gracefully in Python. Your processes might hang for 10 minutes (the NCCL timeout) or become perpetual zombies.
>
> 
> `torchrunx` decouples "launcher" and "worker" processes. If the system kills a worker, our launcher immediately raises a `WorkerFailure` exception, which users can handle as they wish. We always clean up all nodes, so no more zombies!

5. **Bonus features** 🎁

> - Typing for function arguments and return values.
> - Custom, fine-grained handling of logging, environment variables, and exception propagation. We have nice defaults too: no more interleaved logs and irrelevant exceptions!
> - No need to manually set up [`dist.init_process_group`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group)
> - Automatic detection of SLURM environments.
> - Start multi-node training from Python notebooks!

**On our [roadmap](https://github.com/apoorvkh/torchrunx/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement): higher-order parallelism, support for debuggers, and more!**

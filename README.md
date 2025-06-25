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

It enables complex workflows within a single script and has useful features even if only using 1 GPU.

```bash
pip install torchrunx
```

Requires: Linux. If using multiple machines: SSH & shared filesystem.

---

<h4>Example: simple training loop</h4>

Suppose we have some distributed training function (which needs to run on every GPU):

```python
def distributed_training(model: nn.Module, num_steps: int) -> nn.Module: ...
```

<details>
<summary><b>Implementation of <code>distributed_training</code> (click to expand)</b></summary>

```python
from __future__ import annotations
import os
import torch
import torch.nn as nn

def distributed_training(num_steps: int = 10) -> nn.Module | None:
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    model = nn.Linear(10, 10)
    model.to(local_rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters())

    for step in range(num_steps):
        optimizer.zero_grad()

        inputs = torch.randn(5, 10).to(local_rank)
        labels = torch.randn(5, 10).to(local_rank)
        outputs = ddp_model(inputs)

        torch.nn.functional.mse_loss(outputs, labels).backward()
        optimizer.step()

    if rank == 0:
        return model.cpu()
```

</details>

We can distribute and run this function (e.g. on 2 machines x 2 GPUs) using **`torchrunx`**!

```python
import logging
import torchrunx

logging.basicConfig(level=logging.INFO)

launcher = torchrunx.Launcher(
    hostnames = ["localhost", "second_machine"],  # or IP addresses
    workers_per_host = "gpu"  # default, or just: 2
)

results = launcher.run(
    distributed_training,
    num_steps = 10
)
```

Once completed, you can retrieve the results and process them as you wish.

```python
trained_model: nn.Module = results.rank(0)
                     # or: results.index(hostname="localhost", local_rank=0)

# and continue your script
torch.save(trained_model.state_dict(), "outputs/model.pth")
```

**See more examples where we fine-tune LLMs using:**
  - [Transformers](https://torchrun.xyz/examples/transformers.html)
  - [DeepSpeed](https://torchrun.xyz/examples/deepspeed.html)
  - [PyTorch Lightning](https://torchrun.xyz/examples/lightning.html)
  - [Accelerate](https://torchrun.xyz/examples/accelerate.html)

**Refer to our [API](https://torchrun.xyz/api.html) and [Usage](https://torchrun.xyz/usage/general.html) for many more capabilities!**

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

> Your workflow may have steps that are complex (e.g. pre-train, fine-tune, test) or may different parallelizations (e.g. training on 8 GPUs, testing on 1 GPU). In these cases, CLI-based launchers require each step to live in its own script. Our library treats these steps in a modular way, so they can cleanly fit together in a single script!
>
> 
> We clean memory leaks as we go, so previous steps won't crash or adversely affect future steps.

4. **Better handling of system failures. No more zombies!** 🧟

> With `torchrun`, your "work" is inherently coupled to your main Python process. If the system kills one of your workers (e.g. due to RAM OOM or segmentation faults), there is no way to fail gracefully in Python. Your processes might hang for 10 minutes (the NCCL timeout) or become perpetual zombies.
>
> 
> `torchrunx` decouples "launcher" and "worker" processes. If the system kills a worker, our launcher immediately raises a `WorkerFailure` exception, which users can handle as they wish. We always clean up all nodes, so no more zombies!

5. **Bonus features** 🎁

> - Return objects from distributed functions.
> - [Automatic detection of SLURM environments.](https://torchrun.xyz/usage/slurm.html)
> - Start multi-node training from Python notebooks!
> - Our library is fully typed!
> - Custom, fine-grained handling of [logging](https://torchrun.xyz/usage/logging.html), [environment variables](https://torchrun.xyz/usage/general.html#environment-variables), and [exception propagation](https://torchrun.xyz/usage/general.html#exceptions). We have nice defaults too: no more interleaved logs and irrelevant exceptions!

**On our [roadmap](https://github.com/apoorvkh/torchrunx/issues?q=is%3Aopen+is%3Aissue+label%3Aenhancement): higher-order parallelism, support for debuggers, and more!**

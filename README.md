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

**`torchrunx`** is a *functional* utility for distributing PyTorch code across devices. This is a [more convenient, robust, and featureful](https://torchrun.xyz/features.html) alternative to CLI-based launchers, like `torchrun`, `accelerate launch`, and `deepspeed`.

It enables complex workflows within a single script and has useful features even if only using 1 GPU.

```bash
pip install torchrunx
```

Requires: Linux. If using multiple machines: SSH & shared filesystem.

---

<h4>Example: simple training loop</h4>

Suppose we have some distributed training function (which needs to run on every GPU):

```python
from __future__ import annotations
import os
import torch
import torch.nn as nn

def distributed_training(output_dir: str, num_steps: int = 10) -> str | None:
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
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "model.pt")
        torch.save(model, checkpoint_path)
        return checkpoint_path

    return None
```

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
    output_dir = "outputs",
    num_steps = 10,
)
```

Once completed, you can retrieve the results and process them as you wish.

```python
checkpoint_path: str = results.rank(0)
                 # or: results.index(hostname="localhost", local_rank=0)

# and continue your script
model = torch.load(checkpoint_path, weights_only=False)
model.eval()
```

**See more examples where we fine-tune LLMs using:**
  - [Transformers](https://torchrun.xyz/examples/transformers.html)
  - [DeepSpeed](https://torchrun.xyz/examples/deepspeed.html)
  - [PyTorch Lightning](https://torchrun.xyz/examples/lightning.html)
  - [Accelerate](https://torchrun.xyz/examples/accelerate.html)

**Refer to our [API](https://torchrun.xyz/api.html), [Features](https://torchrun.xyz/features.html), and [Usage](https://torchrun.xyz/usage/general.html) for many more capabilities!**

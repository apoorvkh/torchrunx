# Features

### `torchrunx` uniquely offers

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

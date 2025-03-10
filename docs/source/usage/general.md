# General

## Multiple functions in one script

Consider multiple stages of training: pre-training, supervised fine-tuning, RLHF, etc.

Normally, this kind of work is delegated to multiple scripts. Why? Each stage is complicated (prone to memory leaks) and we don't want them to interfere with each other. They may even require different degrees of parallelism.

`torchrunx` solves these problems — even within a single script — by modularizing workloads into isolated, self-cleaning processes.

```python
# 2 nodes x 8 GPUs
train_launcher = torchrunx.Launcher(hostnames=["node1", "node2"], workers_per_host=8)
# 1 GPU
eval_launcher = torchrunx.Launcher(hostnames=["node1"], workers_per_host=1)

# Training & testing

pretrained_model = train_launcher.run(train).rank(0)
pretrained_acc = eval_launcher.run(evaluation, model=pretrained_model).rank(0)
print(f"Pre-trained model accuracy: {pretrained_acc}")

finetuned_model = train_launcher.run(finetuning, model=pretrained_model).rank(0)
finetuned_acc = eval_launcher.run(evaluation, model=finetuned_model).rank(0)
print(f"Fine-tuned model accuracy: {finetuned_acc}")
```

## Exceptions

Exceptions that are raised in workers will be raised by the launcher process. A {mod}`torchrunx.AgentFailedError` or {mod}`torchrunx.WorkerFailedError` will be raised if any agent or worker dies unexpectedly (e.g. if sent a signal from the OS, due to segmentation faults or OOM).

You can catch these errors and handle them as you wish!

```python
for config in configs:  # e.g. hyper-parameter sweep
    try:
        torchrunx.Launcher().run(train, config)
    except torch.cuda.OutOfMemoryError:
        print(f"{config} results in OOM... continuing...")
```

If you are expecting intermittent failures, you can catch errors and invoke retries:

```python
for retry in range(3):
    try:
        torchrunx.Launcher().run(train, resume_from_checkpoint=True)
    except torchrunx.WorkerFailedError as e:
        print(f"Error occurred: {e}")
        print(f"Retrying ({retry}) ...")
    else:  # if run() is successful
        break
```

## Environment variables

Environment variables in the launcher process that pattern match the [``copy_env_vars``](../api.md#torchrunx.Launcher.copy_env_vars) argument are automatically copied to agents and workers. We set useful defaults for Python and PyTorch. You could replace these. Or extend these like:

```python
torchrunx.Launcher(copy_env_vars=(
    torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("HF_HOME", "WANDB_*",)
))
```

You can also pass (1) specific environment variables and values via [``extra_env_vars``](../api.md#torchrunx.Launcher.extra_env_vars) or (2) a ``.env``-style file via [``env_file``](../api.md#torchrunx.Launcher.env_file). Our agents `source {env_file}`.

Finally, we set the following environment variables in each worker: `LOCAL_RANK`, `RANK`, `LOCAL_WORLD_SIZE`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`.

# Using SLURM

Normally, you are expected to provide the `hostnames` argument in {obj}`torchrunx.Launcher` to specify which nodes you would like to launch your function onto.

If your script is running within a SLURM allocation and you set `hostnames` to `"auto"` (default) or `"slurm"`, we will automatically detect the available nodes and distribute your function onto all of these. A {exc}`RuntimeError` will be raised if `hostnames="slurm"` but no SLURM allocation is detected.

## With `sbatch`

You could have a script (`train.py`) that includes:

```python
def distributed_training():
    ...

if __name__ == "__main__":
    torchrunx.Launcher(
        hostnames = "slurm",
        workers_per_host = "gpu"
    ).run(distributed_training)
```

And some `run.batch` file (e.g. allocating 2 nodes with 2 GPUs each):

```bash
#!/bin/bash
#SBATCH --job-name=torchrunx
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2

# TODO: load your virutal environment
python train.py
```

`sbatch run.batch` should then run `python train.py` (the launcher process) on the primary machine in your SLURM allocation. The launcher will automatically distribute the training function onto both allocated nodes (and also parallelize it across the allocated GPUs).

## With `submitit`

If we use the [`submitit`](https://github.com/facebookincubator/submitit) Python library, we can do all of this from a single python script.

```python
def distributed_training():
    ...

def launch_training():
    torchrunx.Launcher(
        hostnames = "slurm",
        workers_per_host = "gpu"
    ).run(distributed_training)

if __name__ == "__main__":
    executor = submitit.SlurmExecutor(folder="slurm_outputs")
    executor.update_parameters(
        use_srun=False, time=60, ntasks_per_node=1,
        nodes=2, gpus_per_node=2
    )
    executor.submit(launch_training)
```

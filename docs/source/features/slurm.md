# SLURM Integration

By default, the `hostnames` or `workers_per_host` arguments are populated from the current SLURM allocation. If no allocation is detected, we assume 1 machine (localhost) with N workers (num. GPUs or CPUs).
Raises a `RuntimeError` if `hostnames="slurm"` or `workers_per_host="slurm"` but no allocation is detected.

# General

## Multiple functions in one script

We could also launch multiple functions (e.g. train on many GPUs, test on one GPU):

```python
import torchrunx as trx

trained_model = trx.launch(
    func=train,
    hostnames=["node1", "node2"],
    workers_per_host=8
).rank(0)

accuracy = trx.launch(
    func=test,
    func_args=(trained_model,),
    hostnames=["localhost"],
    workers_per_host=1
).rank(0)

print(f'Accuracy: {accuracy}')
```

{mod}`torchrunx.launch` is self-cleaning: all processes are terminated (and the used memory is completely released) before the subsequent invocation.

## Retries

Sometimes distributed functions will fail randomly (OOM, networking, or resource errors), and should be executed again. Remember, {mod}`torchrunx.launch` will raise whatever exception its workers raise, so you can catch specific exceptions as you normally would. To retry launching a distributed function, we recommend doing the following:

```python
import torchrunx as trx

n_retries = 5

for r in range(n_retries + 1):
    try:
        trx.launch(train, hostnames=...)
    except CudaOOMError:
        print("retrying")
        if r == n_retries:
            raise Exception("maximum retries attempted")
    else:
        break
    
```

## Environment variables

Environment variables in the launcher process that match the `default_env_vars` argument are automatically copied to agents and workers. We set useful defaults for Python and PyTorch. Environment variables are pattern-matched with this list using `fnmatch`.

`default_env_vars` can be overriden if desired. This list can be augmented using `extra_env_vars`. Additional environment variables (and more custom bash logic) can be included via the `env_file` argument. Our agents `source` this file.

We also set the following environment variables in each worker: `LOCAL_RANK`, `RANK`, `LOCAL_WORLD_SIZE`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`.

## Exceptions

Exceptions that are raised in workers will be raised by the launcher process.

A {mod}`torchrunx.AgentFailedError` or {mod}`torchrunx.WorkerFailedError` will be raised if any agent or worker dies unexpectedly (e.g. if sent a signal from the OS, due to segmentation faults or OOM).

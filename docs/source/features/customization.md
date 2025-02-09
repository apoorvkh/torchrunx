# Customization

## Propagating exceptions

Exceptions that are raised in workers will be raised by the launcher process.

A {mod}`torchrunx.AgentFailedError` or {mod}`torchrunx.WorkerFailedError` will be raised if any agent or worker dies unexpectedly (e.g. if sent a signal from the OS, due to segmentation faults or OOM).

## Environment variables

Environment variables in the launcher process that match the `default_env_vars` argument are automatically copied to agents and workers. We set useful defaults for Python and PyTorch. Environment variables are pattern-matched with this list using `fnmatch`.

`default_env_vars` can be overriden if desired. This list can be augmented using `extra_env_vars`. Additional environment variables (and more custom bash logic) can be included via the `env_file` argument. Our agents `source` this file.

We also set the following environment variables in each worker: `LOCAL_RANK`, `RANK`, `LOCAL_WORLD_SIZE`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`.

## Logging

We forward all logs (i.e. from {mod}`logging` and {mod}`sys.stdout`/{mod}`sys.stderr`) from workers and agents to the launcher. By default, the logs from the first agent and its first worker are printed into the launcher's `stdout` stream. Logs from all agents and workers are written to files in `$TORCHRUNX_LOG_DIR` (default: `./torchrunx_logs`) and are named by timestamp, hostname, and local_rank.

{mod}`logging.Handler` objects can be provided via the `handler_factory` argument to provide further customization (mapping specific agents/workers to custom output streams). You must pass a function that returns a list of {mod}`logging.Handler`s to ``handler_factory``.

We provide some utilities to help:

```{eval-rst}
.. autofunction:: torchrunx.file_handler
```

```{eval-rst}
.. autofunction:: torchrunx.stream_handler
```

```{eval-rst}
.. autofunction:: torchrunx.add_filter_to_handler
```

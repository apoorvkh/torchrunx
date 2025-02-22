# Custom Logging

We forward all logs (i.e. from {mod}`logging` and {mod}`sys.stdout`/{mod}`sys.stderr`) from workers and agents to the launcher. By default, the logs from the first agent and its first worker are printed into the launcher's `stdout` stream. Logs from all agents and workers are written to files in `$TORCHRUNX_LOG_DIR` (default: `./torchrunx_logs`) and are named by timestamp, hostname, and local_rank.

{mod}`logging.Handler` objects can be provided via the `handler_factory` argument to provide further customization (mapping specific agents/workers to custom output streams). You must pass a function that returns a list of {mod}`logging.Handler`s to ``handler_factory``.

We provide some utilities to help:

```{eval-rst}
.. autofunction:: torchrunx.utils.file_handler
```

```{eval-rst}
.. autofunction:: torchrunx.utils.stream_handler
```

```{eval-rst}
.. autofunction:: torchrunx.utils.add_filter_to_handler
```

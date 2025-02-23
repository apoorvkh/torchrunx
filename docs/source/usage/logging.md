# Custom Logging

We forward all worker and agent logs (i.e. from {mod}`logging`, {obj}`sys.stdout`, and {obj}`sys.stderr`) to the launcher for processing.

By default, the logs from the rank 0 agent and worker are printed into the launcher's `stdout` stream. Logs from all agents and workers are written to a directory (by the current timestamp) in `$TORCHRUNX_LOG_DIR` (default: `./torchrunx_logs`).

You can fully customize how logs are processed using {func}`torchrunx.Launcher.set_logging_handlers`. You should provide it a function that constructs and returns a list of {obj}`logging.Handler` objects. Each {obj}`logging.Handler` controls where logs should be written.

We provide some handler utilities that direct a specified worker or agent's logs to a file or stream.

```{eval-rst}
.. autofunction:: torchrunx.utils.file_handler
```

```{eval-rst}
.. autofunction:: torchrunx.utils.stream_handler
```

For example, we could construct and pass a handler factory that streams the rank 0 agent and worker logs to the launcher's `stdout`.

```python
def rank_0_handlers() -> list[logging.Handler]:
    return [
        stream_handler(hostname=hostnames[0], local_rank=None),  # agent 0
        stream_handler(hostname=hostnames[0], local_rank=0),  # worker 0
    ]
```

```python
torchrunx.Launcher(...).set_logging_handlers(rank_0_handlers).run(...)
```

You can also [provide your own ``logging.Handler``](https://docs.python.org/3.9/library/logging.handlers.html#module-logging.handlers) and apply {func}`torchrunx.utils.add_filter_to_handler` to constrain which worker or agent's logs it should process.

```{eval-rst}
.. autofunction:: torchrunx.utils.add_filter_to_handler
```

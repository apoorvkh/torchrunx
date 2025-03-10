# Custom Logging

We forward all agent and worker logs (i.e. from {mod}`logging`, {obj}`sys.stdout`, and {obj}`sys.stderr`) to the launcher process.

## Defaults

By default, the logs from the rank 0 agent and rank 0 worker are handled by loggers on the launcher process (and so they should be printed to `stdout`/`stderr`). You may control these logs like:

```python
logging.basicConfig(level=logging.INFO)
logging.getLogger("torchrunx").setLevel(logging.DEBUG)
logging.getLogger("torchrunx.node1").setLevel(logging.INFO)
logging.getLogger("torchrunx.node1.1").setLevel(logging.INFO)  # worker 1 (local rank) on node 1
```

Also, logs from all agents and workers are written to a directory (by the current timestamp) in `$TORCHRUNX_LOG_DIR` (default: `./torchrunx_logs`). These can be controlled using `$TORCHRUNX_LOG_LEVEL` (default: `INFO`).

## Customization

You can fully customize how logs are processed using {func}`torchrunx.Launcher.set_logging_handlers`. You should provide it a factory function that constructs and returns a list of {obj}`logging.Handler` objects. Each {obj}`logging.Handler` controls where logs should be written. You can also add a filter to restrict the handler to the logs of a specific agent or worker.

Here's an example:

```python
from torchrunx.utils.log_handling import RedirectHandler, get_handler_filter

def custom_handlers() -> list[logging.Handler]:

    # Handler: redirect logs from (host 0, agent) to logger on launcher process
    redirect_handler = RedirectHandler()
    redirect_handler.addFilter(get_handler_filter(
        hostname=hostnames[0], local_rank=None, log_level=logging.DEBUG
    ))

    # Handler: output logs from (host 0, worker 0) to "output.txt"
    file_handler = logging.FileHandler("output.txt")
    file_handler.addFilter(get_handler_filter(
        hostname=hostnames[0], local_rank=0, log_level=logging.DEBUG
    ))

    return [
        redirect_handler,
        file_handler,
    ]
```

```python
torchrunx.Launcher(...).set_logging_handlers(custom_handlers).run(...)
```

Finally, you can control library-specific logging (within the worker processes) by modifying the distributed function:

```python
def distributed_function():
    logging.getLogger("transformers").setLevel(logging.DEBUG)

    logger = logging.getLogger("my_app")
    logger.info("Hello world!")
    ...

torchrunx.Launcher(...).run(distributed_function)
```

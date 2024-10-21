Advanced Usage
==============

Multiple functions in one script
--------------------------------

We could also launch multiple functions (e.g. train on many GPUs, test on one GPU):

.. code-block:: python

    import torchrunx as trx

    trained_model = trx.launch(
        func=train,
        hostnames=["node1", "node2"],
        workers_per_host=8
    ).value(rank=0)

    accuracy = trx.launch(
        func=test,
        func_kwargs={'model': model},
        hostnames=["localhost"],
        workers_per_host=1
    ).value(rank=0)

    print(f'Accuracy: {accuracy}')


``trx.launch()`` is self-cleaning: all processes are terminated (and the used memory is completely released) after each invocation.


SLURM integration
-----------------

By default, the ``hostnames`` or ``workers_per_host`` arguments are populated from the current SLURM allocation. If no allocation is detected, we assume 1 machine (``localhost``) with N GPUs or CPUs.
Raises a ``RuntimeError`` if ``hostnames`` or ``workers_per_host`` are intentionally set to ``"slurm"`` but no allocation is detected.

CLI support
-----------

We provide the :mod:`torchrunx.Launcher` class as an alias to :mod:`torchrunx.launch`.

.. autoclass:: torchrunx.Launcher
    :members:

We can use this class to populate arguments from the CLI (e.g. with `tyro <https://brentyi.github.io/tyro/>`_):

.. code:: python

    import torchrunx as trx
    import tyro

    def distributed_function():
        pass

    if __name__ == "__main__":
        launcher = tyro.cli(trx.Launcher)
        launcher.run(distributed_function)

``python ... --help`` then results in:

.. code:: bash

    ╭─ options ─────────────────────────────────────────────╮
    │ -h, --help           show this help message and exit  │
    │ --hostnames {[STR [STR ...]]}|{auto,slurm}            │
    │                      (default: auto)                  │
    │ --workers-per-host INT|{[INT [INT ...]]}|{auto,slurm} │
    │                      (default: auto)                  │
    │ --ssh-config-file {None}|STR|PATH                     │
    │                      (default: None)                  │
    │ --backend {None,nccl,gloo,mpi,ucc,auto}               │
    │                      (default: auto)                  │
    │ --timeout INT        (default: 600)                   │
    │ --default-env-vars [STR [STR ...]]                    │
    │                      (default: PATH LD_LIBRARY ...)   │
    │ --extra-env-vars [STR [STR ...]]                      │
    │                      (default: )                      │
    │ --env-file {None}|STR|PATH                            │
    │                      (default: None)                  │
    ╰───────────────────────────────────────────────────────╯

Propagating Exceptions
----------------------

Exceptions that are raised in Workers will be raised by the launcher process.

A :mod:`torchrunx.AgentKilledError` will be raised if any agent dies unexpectedly (e.g. if force-killed by the OS, due to segmentation faults or OOM).

Environment Variables
---------------------

Environment variables in the launcher process that match the ``default_env_vars`` argument are automatically copied to agents and workers. We set useful defaults for Python and PyTorch. Environment variables are pattern-matched with this list using ``fnmatch``.

``default_env_vars`` can be overriden if desired. This list can be augmented using ``extra_env_vars``. Additional environment variables (and more custom bash logic) can be included via the ``env_file`` argument. Our agents ``source`` this file.


Custom Logging
--------------

We forward all logs (i.e. from ``logging`` and ``stdio``) from workers and agents to the Launcher. By default, the logs from the first agent and its first worker are printed into the Launcher's ``stdout`` stream. Logs from all agents and workers are written to files in ``$TORCHRUNX_LOG_DIR`` (default: ``./torchrunx_logs``) and are named by timestamp, hostname, and local_rank.

``logging.Handler`` objects can be provided via the ``log_handlers`` argument to provide further customization (mapping specific agents/workers to custom output streams).

We provide some utilities to help:

.. autofunction:: torchrunx.add_filter_to_handler

.. autofunction:: torchrunx.file_handler

.. autofunction:: torchrunx.stream_handler

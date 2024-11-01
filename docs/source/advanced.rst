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
    ).rank(0)

    accuracy = trx.launch(
        func=test,
        func_args=(trained_model,),
        hostnames=["localhost"],
        workers_per_host=1
    ).rank(0)

    print(f'Accuracy: {accuracy}')


:mod:`torchrunx.launch` is self-cleaning: all processes are terminated (and the used memory is completely released) before the subsequent invocation.

Launcher class
--------------

We provide the :mod:`torchrunx.Launcher` class as an alias to :mod:`torchrunx.launch`.

.. autoclass:: torchrunx.Launcher
  :members:

CLI integration
^^^^^^^^^^^^^^^

We can use :mod:`torchrunx.Launcher` to populate arguments from the CLI (e.g. with `tyro <https://brentyi.github.io/tyro/>`_):

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

SLURM integration
-----------------

By default, the ``hostnames`` or ``workers_per_host`` arguments are populated from the current SLURM allocation. If no allocation is detected, we assume 1 machine (localhost) with N workers (num. GPUs or CPUs).
Raises a ``RuntimeError`` if ``hostnames="slurm"`` or ``workers_per_host="slurm"`` but no allocation is detected.

Propagating exceptions
----------------------

Exceptions that are raised in workers will be raised by the launcher process.

A :mod:`torchrunx.AgentFailedError` or :mod:`torchrunx.WorkerFailedError` will be raised if any agent or worker dies unexpectedly (e.g. if sent a signal from the OS, due to segmentation faults or OOM).

Environment variables
---------------------

Environment variables in the launcher process that match the ``default_env_vars`` argument are automatically copied to agents and workers. We set useful defaults for Python and PyTorch. Environment variables are pattern-matched with this list using ``fnmatch``.

``default_env_vars`` can be overriden if desired. This list can be augmented using ``extra_env_vars``. Additional environment variables (and more custom bash logic) can be included via the ``env_file`` argument. Our agents ``source`` this file.

We also set the following environment variables in each worker: ``LOCAL_RANK``, ``RANK``, ``LOCAL_WORLD_SIZE``, ``WORLD_SIZE``, ``MASTER_ADDR``, and ``MASTER_PORT``.

Custom logging
--------------

We forward all logs (i.e. from :mod:`logging` and :mod:`sys.stdout`/:mod:`sys.stderr`) from workers and agents to the launcher. By default, the logs from the first agent and its first worker are printed into the launcher's ``stdout`` stream. Logs from all agents and workers are written to files in ``$TORCHRUNX_LOG_DIR`` (default: ``./torchrunx_logs``) and are named by timestamp, hostname, and local_rank.

:mod:`logging.Handler` objects can be provided via the ``handler_factory`` argument to provide further customization (mapping specific agents/workers to custom output streams). You must pass a function that returns a list of :mod:`logging.Handler`s to ``handler_factory``.

We provide some utilities to help:

.. autofunction:: torchrunx.file_handler

.. autofunction:: torchrunx.stream_handler

.. autofunction:: torchrunx.add_filter_to_handler

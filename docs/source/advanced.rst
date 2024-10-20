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


Environment Detection
---------------------

By default, the `hostnames` or `workers_per_host` :mod:`torchrunx.launch` parameters are set to "auto". These parameters are populated via `SLURM`_ if a SLURM environment is automatically detected. Otherwise, `hostnames = ["localhost"]` and `workers_per_host` is set to the number of GPUs or CPUs (in order of precedence) available locally.

SLURM
+++++

If the `hostnames` or `workers_per_host` parameters are set to `"slurm"`, their values will be filled from the SLURM job. Passing `"slurm"` raises a `RuntimeError` if no SLURM allocation is detected from the environment.

``Launcher`` class
------------------

We provide the ``torchrunx.Launcher`` class as an alternative to ``torchrunx.launch``.

.. autoclass:: torchrunx.Launcher
    :members:
.. .. autofunction:: torchrunx.Launcher.run

CLI Support
+++++++++++

This allows **torchrunx** arguments to be more easily populated by CLI packages like `tyro <https://brentyi.github.io/tyro/>`_:

.. code:: python

    import torchrunx as trx
    import tyro

    def distributed_function():
        print("Hello world!")

    if __name__ == "__main__":
        launcher = tyro.cli(trx.Launcher)
        launcher.run(distributed_function, {})

For example, the `python ... --help` command will then result in:

.. code:: bash

    ╭─ options ─────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ -h, --help              show this help message and exit                                                       │
    │ --hostnames {[STR [STR ...]]}|{auto,slurm}                                                                    │
    │                         (default: auto)                                                                       │
    │ --workers-per-host INT|{[INT [INT ...]]}|{auto,slurm}                                                         │
    │                         (default: auto)                                                                       │
    │ --ssh-config-file {None}|STR|PATH                                                                             │
    │                         (default: None)                                                                       │
    │ --backend {None,nccl,gloo,mpi,ucc,auto}                                                                       │
    │                         (default: auto)                                                                       │
    │ --log-handlers {fixed}  (fixed to: a u t o)                                                                   │
    │ --env-vars STR          (default: PATH LD_LIBRARY LIBRARY_PATH 'PYTHON*' 'CUDA*' 'TORCH*' 'PYTORCH*' 'NCCL*') │
    │ --env-file {None}|STR|PATH                                                                                    │
    │                         (default: None)                                                                       │
    │ --timeout INT           (default: 600)                                                                        │
    ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Custom Logging
--------------

Logs are generated at the worker and agent level, and are specified to :mod:`torchrunx.launch` via the ``log_spec`` argument. By default, a is instantiated, causing logs at the worker and agent levels to be logged to files under ``'./logs'``, and the rank 0 worker's output streams are streamed to the launcher ``stdout``. Logs are prefixed with a timestamp by default. Agent logs have the format ``{timestamp}-{agent hostname}.log`` and workers have the format ``{timestamp}-{agent hostname}[{worker local rank}].log``.

Custom logging classes can be subclassed from the class. Any subclass must have a ``get_map`` method returning a dictionary mapping logger names to lists of :mod:`logging.Handler` objects, in order to be passed to :mod:`torchrunx.launch`. The logger names are of the format ``{agent hostname}`` for agents and ``{agent hostname}[{worker local rank}]`` for workers. The maps all the loggers to :mod:`logging.Filehandler` object pointing to the files mentioned in the previous paragraph. It additionally maps the global rank 0 worker to a :mod:`logging.StreamHandler`, which writes logs the launcher's ``stdout`` stream.

Propagating Exceptions
----------------------

Exceptions that are raised in Workers will be raised in the Launcher process and can be caught by wrapping :mod:`torchrunx.launch` in a try-except clause.

If a worker is killed by the operating system (e.g. due to Segmentation Fault or SIGKILL by running out of memory), the Launcher process raises a RuntimeError.

Environment Variables
---------------------

The :mod:`torchrunx.launch` ``env_vars`` argument allows the user to specify which environmental variables should be copied to the agents from the launcher environment. By default, it attempts to copy variables related to Python and important packages/technologies that **torchrunx** uses such as PyTorch, NCCL, CUDA, and more. Strings provided are matched with the names of environmental variables using ``fnmatch`` - standard UNIX filename pattern matching. The variables are inserted into the agent environments, and then copied to workers' environments when they are spawned.

:mod:`torchrunx.launch` also accepts the ``env_file`` argument, which is designed to expose more advanced environmental configuration to the user. When a file is provided as this argument, the launcher will source the file on each node before executing the agent. This allows for custom bash scripts to be provided in the environmental variables, and allows for node-specific environmental variables to be set.

..
    TODO: example env_file

Support for Numpy >= 2.0
------------------------
only supported if `torch>=2.3`

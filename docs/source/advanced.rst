Advanced Usage
==============

In addition to ``torchrunx.launch``, we provide the ``torchrunx.Launcher`` dataclass. This allows **torchrunx** arguments to be more easily populated by CLI packages like ``tyro``:

.. code:: python

    import torchrunx as trx
    import tyro

    def distributed_function():
        print("Hello world!")

    if __name__ == "__main__":
        launcher = tyro.cli(trx.Launcher)
        launcher.run(distributed_function, {})

.. autoclass:: torchrunx.Launcher
    :members:
.. .. autofunction:: torchrunx.Launcher.run

Logging 
-------

Logs are generated at the worker and agent level, and are specified to :mod:`torchrunx.launch` via the ``log_spec`` argument. By default, a :mod:`torchrunx.DefaultLogSpec` is instantiated, causing logs at the worker and agent levels to be logged to files under ``'./logs'``, and the rank 0 worker's output streams are streamed to the launcher ``stdout``. Logs are prefixed with a timestamp by default. Agent logs have the format ``{timestamp}-{agent hostname}.log`` and workers have the format ``{timestamp}-{agent hostname}[{worker local rank}].log``.

Custom logging classes can be subclassed from the :mod:`torchrunx.LogSpec` class. Any subclass must have a ``get_map`` method returning a dictionary mapping logger names to lists of :mod:`logging.Handler` objects, in order to be passed to :mod:`torchrunx.launch`. The logger names are of the format ``{agent hostname}`` for agents and ``{agent hostname}[{worker local rank}]`` for workers. The :mod:`torchrunx.DefaultLogSpec` maps all the loggers to :mod:`logging.Filehandler` object pointing to the files mentioned in the previous paragraph. It additionally maps the global rank 0 worker to a :mod:`logging.StreamHandler`, which writes logs the launcher's ``stdout`` stream.

Check out the interface of the :mod:`torchrunx.DefaultLogSpec` object below:

.. autoclass:: torchrunx.DefaultLogSpec
    :members:

.. autoclass:: torchrunx.LogSpec
    :members:

.. 
    TODO: example log structure

Worker environment
------------------

The :mod:`torchrunx.launch` ``env_vars`` argument allows the user to specify which environmental variables should be copied to the agents from the launcher environment. By default, it attempts to copy variables related to Python and important packages/technologies that **torchrunx** uses such as PyTorch, NCCL, CUDA, and more. Strings provided are matched with the names of environmental variables using ``fnmatch`` - standard UNIX filename pattern matching. The variables are inserted into the agent environments, and then copied to workers' environments when they are spawned.

:mod:`torchrunx.launch` also accepts the ``env_file`` argument, which is designed to expose more advanced environmental configuration to the user. When a file is provided as this argument, the launcher will source the file on each node before executing the agent. This allows for custom bash scripts to be provided in the environmental variables, and allows for node-specific environmental variables to be set.

..
    TODO: example env_file
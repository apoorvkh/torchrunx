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

.. autofunction:: torchrunx.Launcher.run

Logging 
-------

All logs are generated in the folder provided as the ``logs`` argument to :mod:`torchrunx.launch`. Each worker agent generates a log, named based on the current date and time, followed by the agent hostname. Each worker also has a log, named identically to their agent's log file except for the addition of the worker's local rank at the end of the name. Each agent includes the output from local worker 0 in its log. The launcher renders agent 0's log to ``stdout`` in real time.

.. 
    TODO: example log structure

Worker environment
------------------

The :mod:`torchrunx.launch` ``env_vars`` argument allows the user to specify which evnironmental variables should be copied to the agents from the launcher environment. By default, it attempts to copy variables related to Python and important packages/technologies that **torchrunx** uses such as PyTorch, NCCL, CUDA, and more. Strings provided are matched with the names of environmental variables using ``fnmatch`` - standard UNIX filename pattern matching. The variables are inserted into the agent environments, and then copied to workers' environments when they are spawned.

:mod:`torchrunx.launch` also accepts the ``env_file`` argument, which is designed to expose more advanced environmental configuration to the user. When a file is provided as this argument, the launcher will source the file on each node before executing the agent. This allows for custom bash scripts to be provided in the environmental variables, and allows for node-specific environmental variables to be set.

..
    TODO: example env_file

Numpy >= 2.0
------------
only supported if `torch>=2.3`

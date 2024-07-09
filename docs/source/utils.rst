Utils
=====

.. currentmodule:: torchrunx.utils

**torchrunx** provides some additional utilities that are useful when deploying it.

Slurm 
-----

When using Slurm, the following functions are available to automatically retrieve the hostnames of nodes in an allocation, as well as a suitable number of workers per node. 

.. autofunction:: slurm_hosts

.. autofunction:: slurm_workers

For example, one might populate :mod:`torchrunx.launch`'s ``hostnames`` and ``num_workers`` arguments automatically using these functions:

.. code:: python

    from torchrunx import launch
    from torchrunx.utils import slurm_hosts, slurm_workers

    result = launch(
        ...,
        hostnames=slurm_hosts(),
        num_workers=slurm_workers()
    )

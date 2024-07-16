Contributing
============

Development environment
-----------------------

Ensure you have the latest development environment installed. After cloning our repository, `install pixi <https://pixi.sh/latest/#installation>`_ and run ``pixi shell`` in the repo's root directory. Additionally, we use `ruff <https://github.com/astral-sh/ruff>`_ for linting and formatting, `pyright <https://github.com/microsoft/pyright>`_ for type checking, and ``pytest`` for testing. 

TODO: just write these as Pixi run commands

Testing 
-------

``tests/`` contains ``pytest``-style tests for validating that code changes do not break the core functionality of **torchrunx**. At the moment, we have a few simple CI tests powered by Github action, which are limited to single-agent CPU-only tests due to Github's infrastructure.

Contributing
------------

Make a pull request with your changes and we'll try to look at soon! If addressing a specific issue, mention it in the PR, and offer a short explanation of your fix. If adding a new feature, explain why it's meaningful and belongs in **torchrunx**. 

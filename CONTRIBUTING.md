# Contributing

We use the [`uv`](https://github.com/astral-sh/uv) package manager. Simply [install `uv`](https://github.com/astral-sh/uv#installation) and run `uv sync` in this repository to build the environment. Run `source .venv/bin/activate` to activate the environment.

We use `ruff check` for linting, `ruff format` for formatting, `pyright` for static type checking, and `pytest` for testing.

We build wheels with `uv build` and upload to [PyPI](https://pypi.org/project/torchrunx) with `uv publish`. Our release pipeline is powered by Github Actions.

## Pull Requests

Make a pull request with your changes on Github and we'll try to look at soon! If addressing a specific issue, mention it in the PR, and offer a short explanation of your fix. If adding a new feature, explain why it's meaningful and belongs in __torchrunx__.

## Testing

`tests/` contains `pytest`-style tests for validating that code changes do not break the core functionality of our library.

At the moment, we run `pytest tests/test_ci.py` (i.e. simple single-node CPU-only tests) in our Github Actions CI pipeline (`.github/workflows/release.yml`). One can manually run our more involved tests (on GPUs, on multiple machines from SLURM) on their own hardware.

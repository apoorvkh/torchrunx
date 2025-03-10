mkdir docs/source/artifacts

uv run python -c "from argparse import ArgumentParser; from torchrunx.integrations.parsing import add_torchrunx_argument_group; parser = ArgumentParser(); add_torchrunx_argument_group(parser); parser.parse_args()" --help > docs/source/artifacts/argparse_cli_help.txt
uv run --with tyro python -c "import torchrunx; import tyro; tyro.cli(torchrunx.Launcher)" --help > docs/source/artifacts/tyro_cli_help.txt

uv run --with . scripts/examples/transformers_train.py --help > docs/source/artifacts/transformers_help.txt
uv run --with . scripts/examples/deepspeed_train.py --help > docs/source/artifacts/deepspeed_help.txt
uv run --with . scripts/examples/lightning_train.py --help > docs/source/artifacts/lightning_help.txt
uv run --with . scripts/examples/accelerate_train.py --help > docs/source/artifacts/accelerate_help.txt

mkdir docs/source/artifacts

uv run --with tyro python -c "import torchrunx; import tyro; tyro.cli(torchrunx.Launcher)" --help > docs/source/artifacts/cli_help.txt

uv run scripts/examples/transformers_train.py --help > docs/source/artifacts/transformers_help.txt
uv run scripts/examples/deepspeed_train.py --help > docs/source/artifacts/deepspeed_help.txt
uv run scripts/examples/lightning_train.py --help > docs/source/artifacts/lightning_help.txt
uv run scripts/examples/accelerate_train.py --help > docs/source/artifacts/accelerate_help.txt

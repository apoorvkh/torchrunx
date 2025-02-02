# Transformers

```bash
uv run torchrunx_transformers.py --launcher.hostnames localhost --launcher.workers-per-host 2 --args.output_dir output --args.per-device-train-batch-size 4 --args.report-to tensorboard
```

```{eval-rst}
.. literalinclude:: ./scripts/torchrunx_transformers.py
   :start-after: # [docs:include]
```

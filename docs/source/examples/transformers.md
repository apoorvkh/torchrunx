# Transformers

Here's an example script that uses `torchrunx` with [`transformers.Trainer`](https://huggingface.co/docs/transformers/en/main_classes/trainer) to fine-tune any causal language model (from `transformers`) on any text dataset (from `datasets`) with any number of GPUs or nodes.

[https://torchrun.xyz/transformers_train.py](https://raw.githubusercontent.com/apoorvkh/torchrunx/refs/heads/main/docs/source/examples/scripts/transformers_train.py)

<details>
  <summary><p style="display: inline-block;"><code class="docutils literal notranslate"><span class="pre">python transformers_train.py --help</span></code></p> (expand)</summary>

  ```{eval-rst}
  .. literalinclude:: ../artifacts/transformers_help.txt
  ```
</details>

## Training GPT-2 on WikiText in One Line

The following command runs our script end-to-end: installing all dependencies, downloading model and data, training, logging to TensorBoard, etc. For multi-node training (+ if not using SLURM), you should also pass e.g. `--launcher.hostnames node1 node2`.

Pre-requisite: [uv](https://docs.astral.sh/uv)

```bash
uv run --python "3.12" https://torchrun.xyz/transformers_train.py \
   --model.name gpt2 \
   --dataset.path "Salesforce/wikitext" --dataset.name "wikitext-2-v1" --dataset.split "train" --dataset.num-samples 80 \
   --trainer.output-dir output --trainer.per-device-train-batch-size 4 --trainer.report-to tensorboard
```

You can visualize the logs with:

```bash
uv run --with tensorboard tensorboard --logdir output/runs
```

## Script

```{eval-rst}
.. literalinclude:: ./scripts/transformers_train.py
   :start-after: # [docs:start-after]
```

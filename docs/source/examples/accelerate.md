# Accelerate

Here's an example script that uses `torchrunx` with [Accelerate](https://huggingface.co/docs/accelerate/en/index) to fine-tune any causal language model (from `transformers`) on any text dataset (from `datasets`) with any number of GPUs or nodes.

[https://torchrun.xyz/accelerate_train.py](https://raw.githubusercontent.com/apoorvkh/torchrunx/refs/heads/main/scripts/examples/accelerate_train.py)

<details>
  <summary><p style="display: inline-block;"><code class="docutils literal notranslate"><span class="pre">python accelerate_train.py --help</span></code></p> (expand)</summary>

  ```{eval-rst}
  .. literalinclude:: ../artifacts/accelerate_help.txt
  ```
</details>

## Training GPT-2 on WikiText in One Line

The following command installs dependencies and runs our script (for example, with `GPT-2` on `WikiText`). For multi-node training (+ if not using SLURM), you should also pass e.g. `--launcher.hostnames node1 node2`.

Pre-requisite: [uv](https://docs.astral.sh/uv)

```bash
uv run --python "3.12" https://torchrun.xyz/accelerate_train.py \
   --batch-size 8 --output-dir output \
   --model.name gpt2 \
   --dataset.path "Salesforce/wikitext" --dataset.name "wikitext-2-v1" --dataset.split "train" --dataset.num-samples 80
```

## Script

```{eval-rst}
.. literalinclude:: ./scripts/accelerate_train.py
   :start-after: # [docs:start-after]
```

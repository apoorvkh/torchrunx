# Accelerate

Here's an example script that uses `torchrunx` with [Accelerate](https://huggingface.co/docs/accelerate/en/index) to fine-tune any causal language model (from `transformers`) on any text dataset (from `datasets`) with any number of GPUs or nodes.

[https://torchrun.xyz/accelerate_train.py](https://raw.githubusercontent.com/apoorvkh/torchrunx/refs/heads/main/docs/source/examples/scripts/accelerate_train.py)

<details>
  <summary><p style="display: inline-block;"><code class="docutils literal notranslate"><span class="pre">python accelerate_train.py --help</span></code></p> (expand)</summary>

  ```{eval-rst}
  .. literalinclude:: ./scripts/accelerate_help.txt
  ```
</details>

  - `--launcher`: [torchrunx.Launcher](../api.md#torchrunx.Launcher)
  - `--model`: [`transformers.AutoModelForCausalLM`](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM)
  - `--dataset`: [`datasets.load_dataset`](https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset)

Required: `--model.name`, `--dataset.path`

### Training GPT-2 on WikiText in One Line

The following command runs our script end-to-end: installing all dependencies, downloading model and data, training, logging to TensorBoard, etc. Pre-requisite: [uv](https://docs.astral.sh/uv)

```bash
uv run https://torchrun.xyz/accelerate_train.py \
   --model.name gpt2 \
   --dataset.path "Salesforce/wikitext" --dataset.name "wikitext-2-v1" --dataset.split "train" --dataset.num-samples 80
```

For multi-node training (+ if not using SLURM), you should also pass e.g. `--launcher.hostnames node1 node2`.

### Script

```{eval-rst}
.. literalinclude:: ./scripts/accelerate_train.py
   :start-after: # [docs:start-after]
```

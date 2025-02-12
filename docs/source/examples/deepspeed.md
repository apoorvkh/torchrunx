# DeepSpeed

Here's an example script that uses `torchrunx` with [DeepSpeed](https://www.deepspeed.ai/) to fine-tune any causal language model (from `transformers`) on any text dataset (from `datasets`) with any number of GPUs or nodes.

[https://torchrun.xyz/deepspeed_train.py](https://raw.githubusercontent.com/apoorvkh/torchrunx/refs/heads/main/docs/source/examples/scripts/deepspeed_train.py)

<details>
  <summary><p style="display: inline-block;"><code class="docutils literal notranslate"><span class="pre">python accelerate_train.py --help</span></code></p> (expand)</summary>

  ```{eval-rst}
  .. literalinclude:: ./scripts/deepspeed_help.txt
  ```
</details>

  - `--launcher`: [torchrunx.Launcher](../api.md#torchrunx.Launcher)
  - `--model`: [`transformers.AutoModelForCausalLM`](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM)
  - `--dataset`: [`datasets.load_dataset`](https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset)
  - `--deepspeed`: [`DeepSpeedArgs`](#script)

Required: `--model.name`, `--dataset.path`, `--deepspeed.deepspeed-config`

### Training GPT-2 on WikiText in One Line

The following command runs our script end-to-end: installing all dependencies, downloading model and data, training, logging to TensorBoard, etc. Pre-requisite: [uv](https://docs.astral.sh/uv), and a [DeepSpeed configuration file](#example-configuration-file).

```bash
uv run https://torchrun.xyz/deepspeed_train.py \
   --model.name gpt2 \
   --dataset.path "Salesforce/wikitext" --dataset.name "wikitext-2-v1" --dataset.split "train" --dataset.num-samples 80 \
   --deepspeed.deepspeed-config deepspeed_config.json
```

For multi-node training (+ if not using SLURM), you should also pass e.g. `--launcher.hostnames node1 node2`.

### Script

```{eval-rst}
.. literalinclude:: ./scripts/deepspeed_train.py
   :start-after: # [docs:start-after]
```

### Example configuration file
```{eval-rst}
.. literalinclude:: ./scripts/deepspeed_config.json
```

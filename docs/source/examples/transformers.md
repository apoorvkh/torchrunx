# Transformers

Here's an example script that uses `torchrunx` with [`transformers.Trainer`](https://huggingface.co/docs/transformers/en/main_classes/trainer) to fine-tune any causal language model (from `transformers`) on any text dataset (from `datasets`) with any number of GPUs or nodes: [https://torchrun.xyz/transformers_train.py](https://torchrun.xyz/transformers_train.py).

You can pass command-line arguments to customize:
  - `--launcher`: [torchrunx.Launcher](../api.md#torchrunx.Launcher)
  - `--model`: [`transformers.AutoModelForCausalLM`](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM)
  - `--dataset`: [`transformers.AutoTokenizer`](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer) and [`datasets.load_dataset`](https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset)
  - `--trainer`: [`transformers.TrainingArguments`](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)

The following arguments are required: `--model.name`, `--dataset.tokenizer-name`, `--dataset.path`, `--trainer.output-dir`.

<details>
  <summary><p style="display: inline-block;"><code class="docutils literal notranslate"><span class="pre">python transformers_train.py --help</span></code></p> (expand)</summary>

  ```{eval-rst}
  .. literalinclude:: ./scripts/transformers_help.txt
  ```
</details>

Of course, this script is a template: you can also edit the script first, as desired.

### Training GPT-2 on WikiText in One Line

The following one-line command runs our script end-to-end (installing all dependencies, downloading model and data, training, logging to TensorBoard, etc.).

Pre-requisites: [uv](https://docs.astral.sh/uv)

```bash
uv run https://torchrun.xyz/transformers_train.py \
      --model.name gpt2 --dataset.tokenizer-name gpt2 \
      --dataset.path "Salesforce/wikitext" --dataset.name "wikitext-2-v1" --dataset.split "train" --dataset.num-samples 80 \
      --trainer.output_dir output --trainer.per-device-train-batch-size 4 --trainer.report-to tensorboard
```

We don't need to pass `--launcher` arguments by default. But if you want to do multi-node training (and are not using SLURM), you can also pass e.g. `--launcher.hostnames node1 node2`.

### Script

The [raw source code](https://torchrun.xyz/transformers_train.py) also specifies dependencies at the top of this file — in [PEP 723](https://peps.python.org/pep-0723) format — e.g. for `uv` as above.

```{eval-rst}
.. literalinclude:: ./scripts/transformers_train.py
   :start-after: # [docs:start-after]
```

# DeepSpeed

Here's an example script that uses `torchrunx` with [DeepSpeed](https://www.deepspeed.ai) to fine-tune any causal language model (from `transformers`) on any text dataset (from `datasets`) with any number of GPUs or nodes.

[https://torchrun.xyz/deepspeed_train.py](https://raw.githubusercontent.com/apoorvkh/torchrunx/refs/heads/main/docs/source/examples/scripts/deepspeed_train.py)

<details>
  <summary><p style="display: inline-block;"><code class="docutils literal notranslate"><span class="pre">python deepspeed_train.py --help</span></code></p> (expand)</summary>

  ```{eval-rst}
  .. literalinclude:: ./scripts/deepspeed_help.txt
  ```
</details>

## Training GPT-2 on WikiText

Deepspeed requires additional (non-Python) dependencies. Use the following commands to set up a project. Source: [Apoorv's Blog — Managing Project Dependencies](https://blog.apoorvkh.com/posts/project-dependencies.html)

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Create a project
pixi init my-project --format pyproject
cd my-project

# Install dependencies
pixi project channel add "conda-forge" "nvidia/label/cuda-12.4.0"
pixi add "python=3.12.7" "cuda=12.4.0" "gcc=11.4.0" "gxx=11.4.0"
pixi add --pypi "torch==2.5.1" "deepspeed" "datasets" "tensorboard" "torch" "torchrunx" "transformers" "tyro"

cat <<EOF > .env
export PYTHONNOUSERSITE="1"
export LIBRARY_PATH="\$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib"
export CUDA_HOME="\$CONDA_PREFIX"
EOF

# Activate environment
pixi shell
source .env
```

Download [deepspeed_train.py](https://raw.githubusercontent.com/apoorvkh/torchrunx/refs/heads/main/docs/source/examples/scripts/deepspeed_train.py) and create `deepspeed_config.json` with:

```json
{
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": { "lr": 0.00015 }
    },
    "fp16": { "enabled": true },
    "zero_optimization": true,
    "tensorboard": {
        "enabled": true,
        "output_path": "output/tensorboard/",
        "job_name": "gpt2_wikitext"
    }
}
```

```bash
python deepspeed_train.py --model-name gpt2 --deepspeed-config deepspeed_config.json --checkpoint-dir output \
       --dataset.path "Salesforce/wikitext" --dataset.name "wikitext-2-v1" --dataset.split "train" --dataset.num-samples 80
```

For multi-node training (+ if not using SLURM), you should also pass e.g. `--launcher.hostnames node1 node2`.

You can visualize the logs with:

```bash
tensorboard --logdir output/tensorboard/gpt2_wikitext
```

## Script

```{eval-rst}
.. literalinclude:: ./scripts/deepspeed_train.py
```

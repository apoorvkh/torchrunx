# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "accelerate",
#     "datasets",
#     "torch",
#     "transformers",
#     "tyro",
# ]
# ///

# [docs:start-after]
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

import torchrunx


@dataclass
class ModelConfig:
    name: str


@dataclass
class DatasetConfig:
    path: str
    name: str | None = None
    split: str | None = None
    text_column: str = "text"
    num_samples: int | None = None


def load_training_data(
    tokenizer_name: str,
    dataset_config: DatasetConfig,
) -> Dataset:
    # Load dataset

    dataset = load_dataset(
        dataset_config.path, name=dataset_config.name, split=dataset_config.split
    )
    if dataset_config.num_samples is not None:
        dataset = dataset.select(range(dataset_config.num_samples))

    # Build tokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to suppress warnings
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenize_fn = functools.partial(
        tokenizer,
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding="max_length",
    )

    # Tokenize dataset

    return dataset.map(
        tokenize_fn,
        batched=True,
        input_columns=[dataset_config.text_column],
        remove_columns=[dataset_config.text_column],
    ).map(lambda x: {"labels": x["input_ids"]})


def train(
    model: PreTrainedModel,
    train_dataset: Dataset,
    batch_size: int,
    output_dir: Path,
) -> Path:
    accelerator = Accelerator()

    optimizer = torch.optim.Adam(model.parameters())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        device_batch = {k: torch.stack(v, dim=0).to(accelerator.device) for k, v in batch.items()}
        optimizer.zero_grad()

        loss = model(**device_batch).loss
        print(f"Step {batch_idx}, loss: {loss.item()}", flush=True, end="")
        accelerator.backward(loss)

        optimizer.step()

    accelerator.wait_for_everyone()
    accelerator.save_state(output_dir=output_dir, safe_serialization=False)
    return output_dir / "pytorch_model.bin"


def main(
    launcher: torchrunx.Launcher,
    model_config: Annotated[ModelConfig, tyro.conf.arg(name="model")],
    dataset_config: Annotated[DatasetConfig, tyro.conf.arg(name="dataset")],
    batch_size: int,
    output_dir: Path,
):
    model = AutoModelForCausalLM.from_pretrained(model_config.name)
    train_dataset = load_training_data(tokenizer_name=model_config.name, dataset_config=dataset_config)

    # Launch training
    results = launcher.run(train, (model, train_dataset, batch_size, output_dir))

    # Loading trained model from checkpoint
    checkpoint_path = results.rank(0)
    trained_model = AutoModelForCausalLM.from_pretrained(model_config.name, state_dict=torch.load(checkpoint_path))


if __name__ == "__main__":
    tyro.cli(main)

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "datasets",
#     "deepspeed",
#     "tensorboard",
#     "torch",
#     "torchrunx",
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

import deepspeed
import torch
import tyro
from datasets import load_dataset
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

import torchrunx


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
    deepspeed_config: str | dict,
    checkpoint_dir: str,
) -> None:
    model_engine, _, data_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        config=deepspeed_config,
    )

    model_engine.train()

    for step, batch in enumerate(data_loader):
        input_batch = {k: torch.stack(v).T.to(model_engine.device) for k, v in batch.items()}
        loss = model_engine(**input_batch).loss
        model_engine.backward(loss)
        model_engine.step()
        print(f"Step {step}, loss: {loss.item()}", flush=True, end="")

    model_engine.save_checkpoint(checkpoint_dir)


def main(
    model_name: str,
    deepspeed_config: Path,
    checkpoint_dir: Path,
    dataset_config: Annotated[DatasetConfig, tyro.conf.arg(name="dataset")],
    launcher: torchrunx.Launcher,
):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    train_dataset = load_training_data(tokenizer_name=model_name, dataset_config=dataset_config)

    # Launch training
    launcher.run(train, (model, train_dataset, str(deepspeed_config), str(checkpoint_dir)))

    # Loading trained model from checkpoint
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
    trained_model = AutoModelForCausalLM.from_pretrained(model_name)
    trained_model.load_state_dict(state_dict)


if __name__ == "__main__":
    tyro.cli(main)

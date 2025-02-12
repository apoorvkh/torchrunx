# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "deepspeed",
#     "datasets",
#     "tensorboard",
#     "torch",
#     "torchrunx",
#     "transformers",
#     "tyro",
# ]
# ///

import argparse
import functools
import os
from dataclasses import dataclass
from typing import Annotated

import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import torch

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, AutoConfig

import torchrunx
import tyro


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


@dataclass
class DeepSpeedArgs:
    deepspeed_config: str
    local_rank: int | None = None


def load_training_data(
    tokenizer_name: str,
    dataset_config: DatasetConfig,
) -> Dataset:
    # Load dataset

    dataset = load_dataset(dataset_config.path, name=dataset_config.name, split=dataset_config.split)
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
    deepspeed_args: DeepSpeedArgs
) -> str:

    deepspeed_args.local_rank = int(os.environ["LOCAL_RANK"])

    model_engine, _, loader, _ = deepspeed.initialize(
        args=deepspeed_args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset
    )

    model_engine.train()
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 10:
            break
        device_batch = {k: torch.stack(v, dim=0).to(model_engine.device) for k, v in batch.items()}
        model_engine.zero_grad()

        loss = model_engine(**device_batch).loss
        print(f"Step {batch_idx}, loss: {loss.item()}", flush=True, end="")
        model_engine.backward(loss)

        model_engine.step()

    checkpoint_dir = "output"
    model_engine.save_checkpoint(checkpoint_dir)

    return checkpoint_dir

def main(
    launcher: torchrunx.Launcher,
    model_config: Annotated[ModelConfig, tyro.conf.arg(name="model")],
    dataset_config: Annotated[DatasetConfig, tyro.conf.arg(name="dataset")],
    deepspeed_args: Annotated[DeepSpeedArgs, tyro.conf.arg(name="deepspeed")]
):
    model = AutoModelForCausalLM.from_pretrained(model_config.name)
    train_dataset = load_training_data(tokenizer_name=model_config.name, dataset_config=dataset_config)

    # Launch training
    results = launcher.run(train, (model, train_dataset, deepspeed_args))

    # Loading trained model from checkpoint
    checkpoint_path = results.rank(0)
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
    trained_model = AutoModelForCausalLM.from_config(
        AutoConfig.from_pretrained(model_config.name)
    )
    trained_model.load_state_dict(state_dict)


if __name__ == "__main__":
    tyro.cli(main)

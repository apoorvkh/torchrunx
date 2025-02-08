# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "tensorboard",
#     "torchrunx",
#     "transformers[torch]",
#     "tyro",
# ]
# ///

# [docs:start-after]
import functools
import os
from dataclasses import dataclass
from typing import Annotated

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    trainer_utils,
)
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
    training_args: TrainingArguments,
) -> str:
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
    )

    trainer.train()

    return trainer_utils.get_last_checkpoint(training_args.output_dir)


def main(
    launcher: torchrunx.Launcher,
    model_config: Annotated[ModelConfig, tyro.conf.arg(name="model")],
    dataset_config: Annotated[DatasetConfig, tyro.conf.arg(name="dataset")],
    training_args: Annotated[TrainingArguments, tyro.conf.arg(name="trainer", help="")],
):
    model = AutoModelForCausalLM.from_pretrained(model_config.name)
    train_dataset = load_training_data(tokenizer_name=model_config.name, dataset_config=dataset_config)

    # Launch training
    results = launcher.run(train, (model, train_dataset, training_args))

    # Loading trained model from checkpoint
    checkpoint_path = results.rank(0)
    trained_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)


if __name__ == "__main__":
    tyro.cli(main)

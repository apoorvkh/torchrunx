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
from typing import Annotated

import tyro
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

import torchrunx


def build_model(name: str) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(name)


def load_training_data(
    tokenizer_name: str,
    path: str,
    name: str | None = None,
    split: str | None = None,
    text_column_name: str = "text",
    num_samples: int | None = None,
) -> Dataset:
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

    dataset = load_dataset(path, name=name, split=split)

    if num_samples is None:
        num_samples = len(dataset)

    return (
        dataset.select(range(num_samples))
        .map(
            tokenize_fn,
            batched=True,
            input_columns=[text_column_name],
            remove_columns=[text_column_name],
        )
        .map(lambda x: {"labels": x["input_ids"]})
    )


def train(
    model: PreTrainedModel, training_args: TrainingArguments, train_dataset: Dataset
) -> PreTrainedModel | None:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # TODO: return checkpoint path
    if int(os.environ["RANK"]) == 0:
        return model


def main(
    launcher: torchrunx.Launcher,
    model: Annotated[PreTrainedModel, tyro.conf.arg(prefix_name=False, constructor=build_model)],
    train_dataset: Annotated[Dataset, tyro.conf.arg(name="dataset", constructor=load_training_data)],
    training_args: Annotated[TrainingArguments, tyro.conf.arg(name="trainer", help="")],
):
    results = launcher.run(train, (model, training_args, train_dataset))
    model = results.rank(0)


if __name__ == "__main__":
    tyro.cli(main)

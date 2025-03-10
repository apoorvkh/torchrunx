# /// script
# requires-python = ">=3.9"
# dependencies = [
#    "datasets",
#     "lightning",
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
from typing import Annotated

import lightning as L
import torch

import tyro
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

import torchrunx
from torchrunx.integrations.lightning import TorchrunxClusterEnvironment


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


class CausalLMLightningWrapper(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, *args):  # pyright: ignore
        device_batch = {k: torch.stack(v, dim=0).to(self.model.device) for k, v in batch.items()}
        loss = self.model(**device_batch).loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


def train(model: PreTrainedModel, train_dataset: Dataset) -> str:
    lightning_model = CausalLMLightningWrapper(model)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=1,
        strategy="ddp",
        plugins=[TorchrunxClusterEnvironment()],
        enable_checkpointing=False,
    )

    trainer.fit(model=lightning_model, train_dataloaders=train_loader)
    checkpoint = f"{trainer.log_dir}/final.ckpt"
    trainer.save_checkpoint(checkpoint)

    return checkpoint


def main(
    launcher: torchrunx.Launcher,
    model_config: Annotated[ModelConfig, tyro.conf.arg(name="model")],
    dataset_config: Annotated[DatasetConfig, tyro.conf.arg(name="dataset")],
):
    model = AutoModelForCausalLM.from_pretrained(model_config.name)
    train_dataset = load_training_data(
        tokenizer_name=model_config.name, dataset_config=dataset_config
    )

    # Launch training
    results = launcher.run(train, (model, train_dataset))

    # Loading trained model from checkpoint
    checkpoint_path = results.rank(0)
    dummy_model = AutoModelForCausalLM.from_pretrained(model_config.name)
    trained_model = CausalLMLightningWrapper(dummy_model)
    trained_model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    trained_model = trained_model.model


if __name__ == "__main__":
    tyro.cli(main)

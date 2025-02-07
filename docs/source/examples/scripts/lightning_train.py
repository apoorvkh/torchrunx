import os
from pathlib import Path

import lightning as L
import torch
from datasets import load_dataset

from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import torchrunx
from torchrunx.integrations.lightning import TorchrunxClusterEnvironment

class GPT2CausalLMDataset(Dataset):
    def __init__(self, text_dataset):
        self.dataset = text_dataset
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = 1024

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.dataset[idx]["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded.input_ids.squeeze()
        attention_mask = encoded.attention_mask.squeeze()
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class GPT2LightningWrapper(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def training_step(self, batch, *args): # pyright: ignore
        device_batch = {k: v.to(self.model.device) for k, v in batch.items()}
        loss = self.model(**device_batch).loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


def train():
    lightning_model = GPT2LightningWrapper()

    wikitext_train = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    train_dataset = GPT2CausalLMDataset(text_dataset=wikitext_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

    trainer = L.Trainer(
        accelerator="gpu",
        limit_train_batches=10,
        max_epochs=1,
        devices=2,
        num_nodes=1,
        strategy="ddp",
        plugins=[TorchrunxClusterEnvironment()],
        enable_checkpointing=False
    )

    trainer.fit(model=lightning_model, train_dataloaders=train_loader)
    checkpoint  = f"{trainer.log_dir}/final.ckpt"
    trainer.save_checkpoint(checkpoint)

    return checkpoint


if __name__ == "__main__":
    results = torchrunx.launch(
        func=train,
        hostnames=["localhost"],
        workers_per_host=2,
    )

    checkpoint_path = results.rank(0)
    print(f"Checkpoint at: {checkpoint_path}")

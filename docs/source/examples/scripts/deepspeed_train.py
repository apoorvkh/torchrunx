from dataclasses import dataclass
from pathlib import Path

import deepspeed
import torch

from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import torchrunx


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


@dataclass
class DSPArgs:
    deepspeed_config: str
    # train_batch_size: int
    # batch_size: int


def train():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # optimizer = torch.optim.Adam(model.parameters())
    wikitext_train = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    train_dataset = GPT2CausalLMDataset(text_dataset=wikitext_train)

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=DSPArgs(deepspeed_config="dsp_config.json"),
        model=model,
        model_parameters=model.parameters(),
    )

    model.train()
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 10:
            break
        print(f"Step {batch_idx}")

        device_batch = {k: v.to(model.device) for k, v in batch.items()}

        model.zero_grad()

        loss = model_engine(**device_batch).loss
        model_engine.backward(loss)

        model_engine.step()


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    results = torchrunx.launch(
        func=train,
        hostnames=["localhost"],
        workers_per_host=1,
    )

from pathlib import Path

import torch
from accelerate import Accelerator
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


def train():
    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    optimizer = torch.optim.Adam(model.parameters())
    wikitext_train = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    train_dataset = GPT2CausalLMDataset(text_dataset=wikitext_train)

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    model.train()
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 10:
            break
        print(f"Step {batch_idx}")
        device_batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        optimizer.zero_grad()

        loss = model(**device_batch).loss
        accelerator.backward(loss)

        optimizer.step()

    return model


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    results = torchrunx.launch(
        func=train,
        hostnames=["localhost"],
        workers_per_host=2,
    )

    trained_model: nn.Module = results.rank(0)
    torch.save(trained_model.state_dict(), "output/model.pth")

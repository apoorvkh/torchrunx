# Examples

## Training GPT-2 on WikiText

We will show examples of how to use `torchrunx` alongside several deep learning libraries to train a GPT-2 (small) model with text data from WikiText.

### Accelerate

```python
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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
```

```python
import torchrunx

if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    results = torchrunx.launch(
        func=train,
        hostnames=["localhost"],
        workers_per_host=1,
    )

    trained_model: nn.Module = results.rank(0)
    torch.save(trained_model.state_dict(), "output/model.pth")
```

### HF Trainer

```python
from __future__ import annotations

import os

from datasets import Dataset, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)


def build_model() -> PreTrainedModel:
    config = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_config(config)
    return model


def load_training_data() -> Dataset:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to suppress warnings
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    return (
        load_dataset("Salesforce/wikitext", name="wikitext-2-v1", split="train")
        .select(range(8))
        .map(
            lambda x: tokenizer(
                x["text"],
                max_length=1024,
                truncation=True,
                padding="max_length",
            ),
            batched=True,
            remove_columns=["text"],
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

    if int(os.environ["RANK"]) == 0:
        return model
```

```python
import torchrunx


if __name__ == "__main__":
    model = build_model()
    training_args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=2,
        report_to="tensorboard",
    )
    train_dataset = load_training_data()

    results = torchrunx.launch(
        func=train,
        func_args=(model, training_args, train_dataset),
        hostnames=["localhost"],
        workers_per_host=2,
    )

    model = results.rank(0)
```

### DeepSpeed

### PyTorch Lightning

### MosaicML Composer

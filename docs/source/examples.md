# Examples

## Using `torchrunx` with other deep learning libraries

We will show examples of how to use `torchrunx` to train a GPT-2 (small) with text data from wikitext.

### Accelerate

### HF Trainer

```python
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


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
        labels = input_ids.clone()

        return {"input_ids": input_ids, "labels": labels}


def train():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    wikitext_train = load_dataset(
        "Salesforce/wikitext", name="wikitext-2-v1", split="train"
    )
    train_dataset = GPT2CausalLMDataset(text_dataset=wikitext_train)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="output",
            per_device_train_batch_size=16,
            max_steps=10,
        ),
        train_dataset=train_dataset,
    )

    trainer.train()

    return model
```

```python
import torchrunx

if __name__ == "__main__":
    results = torchrunx.launch(
        func=train,
        hostnames=["localhost"],
        workers_per_host=1,
    )

    trained_model: nn.Module = results.rank(0)
    torch.save(trained_model.state_dict(), "output/model.pth")
```

### DeepSpeed

### PyTorch Lightning

### MosaicML Composer

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "tensorboard",
#     "torchrunx",
#     "transformers[torch]",
# ]
# ///

# [docs:include]
import os
import torchrunx

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

import copy

import submitit
import torch
from torch.utils.data import Dataset
from transformers import BertForMaskedLM, Trainer, TrainingArguments

import torchrunx as trx


class DummyDataset(Dataset):
    def __init__(self, max_text_length=16, num_samples=20000) -> None:
        super().__init__()
        self.input_ids = torch.randint(0, 30522, (num_samples, max_text_length))
        self.labels = copy.deepcopy(self.input_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
        }


def main():
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    train_dataset = DummyDataset()

    ## Training

    training_arguments = TrainingArguments(
        output_dir="output",
        do_train=True,
        per_device_train_batch_size=16,
        max_steps=20,
    )

    trainer = Trainer(
        model=model,  # type: ignore
        args=training_arguments,
        train_dataset=train_dataset,
    )

    trainer.train()


def launch():
    trx.launch(func=main, func_kwargs={}, hostnames="slurm", workers_per_host="slurm")


def test_submitit():
    executor = submitit.SlurmExecutor(folder="logs")

    executor.update_parameters(
        time=60,
        nodes=1,
        ntasks_per_node=1,
        mem="32G",
        cpus_per_task=4,
        gpus_per_node=2,
        constraint="geforce3090",
        partition="3090-gcondo",
        stderr_to_stdout=True,
        use_srun=False,
    )

    executor.submit(launch).result()


if __name__ == "__main__":
    executor = submitit.SlurmExecutor(folder="logs")

    executor.update_parameters(
        time=60,
        nodes=1,
        ntasks_per_node=1,
        mem="32G",
        cpus_per_task=4,
        gpus_per_node=2,
        constraint="geforce3090",
        partition="3090-gcondo",
        stderr_to_stdout=True,
        use_srun=False,
    )

    executor.submit(launch)

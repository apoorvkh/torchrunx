# torchrunx

Example code

```python
def find_max_batch_size_worker(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
) -> int:
    hf_training_args = training_args.to_huggingface(cls=MaxBatchSizeArguments)
    trainer = build_trainer(
        data_args=data_args,
        model_args=model_args,
        hf_training_args=hf_training_args,
    )
    trainer.accelerator.free_memory = lambda *args: None
    return find_max_batch_size(trainer)


@step(bind=True, cacheable=True, version="001")
def find_max_batch_size_step(
    self,
    system_specs: SystemSpecifications,
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
) -> int:
    return run_distributed(
        find_max_batch_size_worker,
        args=(data_args, model_args, training_args),
        # log_dir=self.work_dir,
        num_nodes=system_specs.num_nodes,
        num_procs=system_specs.gpus_per_node,
    )[0]
```

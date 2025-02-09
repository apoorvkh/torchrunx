# Workflows

## Multiple functions in one script

We could also launch multiple functions (e.g. train on many GPUs, test on one GPU):

```python
import torchrunx as trx

trained_model = trx.launch(
    func=train,
    hostnames=["node1", "node2"],
    workers_per_host=8
).rank(0)

accuracy = trx.launch(
    func=test,
    func_args=(trained_model,),
    hostnames=["localhost"],
    workers_per_host=1
).rank(0)

print(f'Accuracy: {accuracy}')
```

{mod}`torchrunx.launch` is self-cleaning: all processes are terminated (and the used memory is completely released) before the subsequent invocation.

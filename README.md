# torchrunx

## Installation

For development:

1. [Install pixi](https://pixi.sh/latest/#installation)
2. `pixi install`
3. `pixi shell`

## Testing/Examples

1. Allocate an interactive slurm allocation. The number of nodes and tasks per node can be adjusted here.
    ```
    salloc --time 01:00:00 -J interact -p batch --nodes 2 \
    --ntasks-per-node 2 --mem 4G --cpus-per-task 1
    ```
2. Activate the development environment.
    ```
    pixi shell
    ```
3. Run the proof-of-concept example script. The script will run an example function using collective functions on each node in the slurm allocation. A total of $n \cdot t$ workers will be spawned for $n$ nodes and $t$ tasks per node, as allocated in the previous step.
    ```
    python3 examples/slurm_poc.py
    ```
    If successful, "PASS" will be printed, as well as the jointly computed tensor from the example function.

### Sample output:
```
torchrunx-3.9[pcurtin1@vscode1 torchrunx]$ salloc --time 01:00:00 -J interact -p batch --nodes 2 \
    --ntasks-per-node 2 --mem 4G --cpus-per-task 1
salloc: Pending job allocation 2615868
salloc: job 2615868 queued and waiting for resources
salloc: job 2615868 has been allocated resources
salloc: Granted job allocation 2615868
salloc: Waiting for resource configuration
salloc: Nodes node[1320,1322] are ready for job
[pcurtin1@node1320 torchrunx]$ pixi shell
 . "/tmp/pixi_env_fli.sh"
[pcurtin1@node1320 torchrunx]$  . "/tmp/pixi_env_fli.sh"
(torchrunx) [pcurtin1@node1320 torchrunx]$ python3 examples/slurm_poc.py
tensor([[ 98.6374, 105.6468, 107.4097,  ...,  90.8607,  98.8783, 103.7635],
        [103.9656, 111.9795, 114.9383,  ...,  99.9243, 104.9417, 109.3429],
        [101.6292, 105.4917, 107.5026,  ...,  89.4010, 100.7006, 106.5736],
        ...,
        [101.1215, 102.2009, 105.3939,  ...,  84.7023,  96.8405, 103.0081],
        [101.6793, 106.1848, 109.0237,  ...,  90.9153, 100.8405, 105.2752],
        [ 99.1340, 107.7936, 104.1491,  ...,  89.6584,  99.2947, 102.3622]])
PASS
(torchrunx) [pcurtin1@node1320 torchrunx]$
```
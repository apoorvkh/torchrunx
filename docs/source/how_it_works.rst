How it works
============

If you want to (e.g.) train your model on several machines with **N** GPUs each, you should run your training function in **N** parallel processes on each machine. During training, each of these processes runs the same training code (i.e. your function) and communicate with each other (e.g. to synchronize gradients) using a `distributed process group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`_.

Your script can call our library (via `mod:torchrunx.launch`) and specify a function to distribute. The main process running your script is henceforth known as the **launcher** process.

Our launcher process spawns an **agent** process (via SSH) on each machine. Each agent then spawns **N** processes (known as **workers**) on its machine. All workers form a process group (with the specified `mod:torchrunx.launch` ``backend``) and run your function in parallel.

**Agent–Worker Communication.** Our agents poll their workers every second and time-out if unresponsive for 5 seconds. Upon polling, our agents receive ``None`` (if the worker is still running) or a `RunProcsResult <https://pytorch.org/docs/stable/elastic/multiprocessing.html#torch.distributed.elastic.multiprocessing.api.RunProcsResult>`_, indicating that the workers have either completed (providing an object returned from or the exception raised by our function) or failed (e.g. due to segmentation fault or OS signal).

**Launcher–Agent Communication.** The launcher and agents form a distributed group (with the CPU-based `GLOO backend <https://pytorch.org/docs/stable/distributed.html#backends>`_) for the communication purposes of our library. Our agents synchronize their own "statuses" with each other and the launcher. An agent's status can include whether it is running/failed/completed and the result of the function. If the launcher or any agent fails to synchronize, all raise a `mod:torchrunx.AgentFailedError` and terminate. If any worker fails or raises an exception, the launcher raises a `mod:torchrunx.WorkerFailedError` or that exception and terminates along with all the agents. If all agents succeed, the launcher returns the objects returned by each worker.

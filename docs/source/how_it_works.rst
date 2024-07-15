How it works
============

In order to organize processes on different nodes, **torchrunx** maintains the following hierarchy:

#. The launcher, the process in which ``torchrunx.Launcher.run`` is executed: Connects to remote hosts and initializes and configures "agents", passes errors and return values from agents to the caller, and is responsible for cleaning up.
#. The agents, initialized on machines where computation is to be performed: Responsible for starting and monitoring "workers".
#. The workers, spawned by agents: Responsible for initializing a ``torch.distributed`` process group, and running the distributed function provided by the user.

An example of how this hierarchy might look in practice is the following: 
Suppose we wish to distribute a training function over four GPUs, and we have access to a cluster where nodes have two available GPUs each. Say that a single instance of our training function can leverage multiple GPUs. We can choose two available nodes and use the launcher to launch our function on those two nodes, specifying that we only need one worker per node, since a single instance of our training function can use both GPUs on each node. The launcher will launch an agent on each node and pass our configuration to the agents, after which the agents will each initialize one worker to begin executing the training function. We could also run two workers per node, each with one GPU, giving us four workers, although this would be slower. 

The launcher initializes the agents by simply SSHing into the provided hosts, and executing our agent code there. The launcher also provides key environmental variables from the launch environment to the sessions where the agents are started and tries to activate the same Python environment that was used to execute the launcher. This is one reason why all machines either running a launcher or agent process should share a filesystem.

The launcher and agents perform exception handling such that any exceptions in the worker processes are appropriately raised by the launcher process. The launcher and agents communicate using a ``torch.distributed`` process group, separate from the group that the workers use. 
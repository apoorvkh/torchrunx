from functools import partial

import os
import socket
import random
from typing import List, Tuple

from multiprocessing import Process, Queue

import dill
import paramiko


def my_function(x, y):
    return x + y

serialized_function = dill.dumps(my_function)

def spawnx(config, func, **kwargs):
  func_to_serialize = partial(func, **kwargs)

  hostname = socket.gethostname()
  ip_address = socket.gethostbyname(hostname)

  print(f"Hostname: {hostname}")
  print(f"IP Address: {ip_address}") 

  open_ports = find_open_ports(config.master_ip, config.master_port_range)
  print(f"Open ports on {config.master_ip}: {open_ports}")
  # let's just choose a random one for now

  port_use = random.choice(open_ports)

  for i, (ip_forgn, port_forgn, user) in enumerate(config.ips_port_users):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
    # so do not need to manually add

    client.connect(ip_forgn, port_forgn, user) 
    # potentially need either pub key or user name


    # stdin, stdout, stderr = client.exec_command(f'echo "{serialized_function}" > my_function.pkl')
    stdin, stdout, stderr = client.exec_command(
        f'python -m torchrunx "{serialized_function}" {config.num_nodes} {config.num_processes} {ip_address} {port_use} {i}'
        )
    print(stdout.read().decode())
    client.close()


def torchrunx(serialized_func, num_nodes, num_processes, ip_master, port_master, rank):

  deserialized_func = dill.loads(serialized_function)
  def worker(func, args): # use nccl/gloo for workers
    result = func(*args)
    queue.put(result)

  queue = Queue()

  # Create and start the processes
  processes = []
  for i in range(num_processes):
      p = Process(target=worker, args=(deserialized_func, (i, i+1)))
      processes.append(p)
      p.start()

  for p in processes:
      p.join()

  pass

# no gloo, nccl in python, just mpi4py 

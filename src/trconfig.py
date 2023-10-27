from dataclasses import dataclass

from typing import List, Tuple

@dataclass
class TorchrunxConfig: 
  num_nodes: int = 4
  num_processes: int = 4  # per node
  timeout: int = 300    
  max_retries: int = 3
  master_ip : str = '127.0.0.1'
  master_port_range : Tuple[int, int] = (20, 1024)
  log_file: str = 'parallel_processing.log'
  ips_port_users: List[Tuple[str, int, str]] = None
  messaging : str = "gloo"

  def __post_init__(self):
        if self.ips_port_users is None:
            self.ips_port_users = []

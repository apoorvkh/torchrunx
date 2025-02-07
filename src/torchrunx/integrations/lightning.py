"""Pytorch Lightning extension utilities."""

import torch
from lightning.fabric.plugins.environments.torchelastic import (  # pyright: ignore [reportMissingImports]
    TorchElasticEnvironment,
)


class TorchrunxClusterEnvironment(TorchElasticEnvironment):
    """PyTorch Lightning ClusterEnvironment compatible with torchrunx."""

    @staticmethod
    def detect() -> bool:
        """Returns ``True`` if the current process was launched using torchrunx."""
        return torch.distributed.is_available()

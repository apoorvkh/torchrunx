"""Integration with PyTorch Lightning Trainer."""

from lightning.fabric.plugins.environments.torchelastic import (  # pyright: ignore [reportMissingImports]
    TorchElasticEnvironment,
)


class TorchrunxClusterEnvironment(TorchElasticEnvironment):
    """Compatible ClusterEnvironment for PyTorch Lightning."""

    @staticmethod
    def detect() -> bool:
        """Force use of the TorchElasticEnvironment."""
        return True

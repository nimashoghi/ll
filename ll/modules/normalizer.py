import torch

from ..config import TypedConfig


class NormalizerConfig(TypedConfig):
    enabled: bool = True

    mean: float = 0.0
    std: float = 1.0

    def normalize(self, x: torch.Tensor):
        if not self.enabled:
            return x
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor):
        if not self.enabled:
            return x
        return x * self.std + self.mean

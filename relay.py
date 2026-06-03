"""Latent read/write components shared by relay experiments."""

import torch
import torch.nn.functional as F
from torch import nn


class LatentReader(nn.Module):
    """Project a hidden activation onto a fixed unit direction."""

    def __init__(self, hidden_size: int, seed: int = 0) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        generator = torch.Generator().manual_seed(seed)
        direction = F.normalize(torch.randn(hidden_size, generator=generator), dim=0)
        self.register_buffer("direction", direction)

    def read_state(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return z in [-1, 1], preserving all dimensions before hidden_size."""
        if hidden.shape[-1] != self.direction.numel():
            raise ValueError(
                f"expected hidden size {self.direction.numel()}, got {hidden.shape[-1]}"
            )
        return torch.tanh(hidden.to(self.direction) @ self.direction)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.read_state(hidden)

"""Latent read/write components shared by relay experiments."""

import torch
import torch.nn.functional as F
from torch import nn


def fit_direction(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Fit a unit direction from the difference between two activation sets."""
    if positive.ndim < 2 or negative.ndim < 2:
        raise ValueError("activation sets must have a batch dimension")
    if positive.shape[1:] != negative.shape[1:]:
        raise ValueError("activation shapes must match after the batch dimension")
    direction = positive.float().mean(dim=0) - negative.float().mean(dim=0)
    if direction.ndim != 1:
        raise ValueError("fit_direction expects final-token activations")
    if direction.norm() == 0:
        raise ValueError("positive and negative activations produce no direction")
    return F.normalize(direction, dim=0)


class LatentReader(nn.Module):
    """Measure a hidden activation against a fixed unit direction."""

    def __init__(self, hidden_size: int, seed: int = 0) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        generator = torch.Generator().manual_seed(seed)
        direction = F.normalize(torch.randn(hidden_size, generator=generator), dim=0)
        self.register_buffer("direction", direction)

    @classmethod
    def from_direction(cls, direction: torch.Tensor) -> "LatentReader":
        """Create a reader from a learned or otherwise supplied direction."""
        if direction.ndim != 1 or direction.numel() < 1:
            raise ValueError("direction must be a non-empty vector")
        if direction.norm() == 0:
            raise ValueError("direction must be non-zero")
        reader = cls.__new__(cls)
        nn.Module.__init__(reader)
        reader.register_buffer("direction", F.normalize(direction.float(), dim=0))
        return reader

    def read_state(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return z in [-1, 1], preserving all dimensions before hidden_size."""
        if hidden.shape[-1] != self.direction.numel():
            raise ValueError(
                f"expected hidden size {self.direction.numel()}, got {hidden.shape[-1]}"
            )
        normalized = F.normalize(hidden, dim=-1)
        return torch.tanh(normalized @ self.direction.to(hidden))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.read_state(hidden)


class LatentState:
    """A scalar state that persists between independent forward passes."""

    def __init__(self, beta: float = 0.0) -> None:
        if not 0 <= beta <= 1:
            raise ValueError("beta must be between 0 and 1")
        self.beta = beta
        self.z = 0.0

    def update(self, observed: float | torch.Tensor) -> float:
        value = float(torch.as_tensor(observed).detach())
        self.z = self.beta * self.z + (1 - self.beta) * value
        return self.z

    def reset(self) -> None:
        self.z = 0.0


class LatentWriter(nn.Module):
    """Add a scalar latent state along a fixed unit direction."""

    def __init__(self, hidden_size: int, seed: int = 1) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        generator = torch.Generator().manual_seed(seed)
        direction = F.normalize(torch.randn(hidden_size, generator=generator), dim=0)
        self.register_buffer("direction", direction)

    def write_state(
        self, hidden: torch.Tensor, z: float | torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        if hidden.shape[-1] != self.direction.numel():
            raise ValueError(
                f"expected hidden size {self.direction.numel()}, got {hidden.shape[-1]}"
            )
        state = torch.as_tensor(z, device=hidden.device, dtype=hidden.dtype)
        return hidden + alpha * state * self.direction.to(hidden)

    def forward(
        self, hidden: torch.Tensor, z: float | torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        return self.write_state(hidden, z, alpha)

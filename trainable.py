"""Trainable low-rank read, update, and write maps around a frozen transformer."""

import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class TrainableRelay(nn.Module):
    """z = beta*z_prev + (1-beta)*tanh(U h); h' = h + alpha*V z."""

    def __init__(self, hidden_size: int, latent_size: int = 4, seed: int = 0) -> None:
        super().__init__()
        if hidden_size < 1 or latent_size < 1:
            raise ValueError("hidden_size and latent_size must be positive")
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        # Keep initialization reproducible without altering the caller's RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.read = nn.Linear(hidden_size, latent_size)
            self.write = nn.Linear(latent_size, hidden_size, bias=False)
            nn.init.normal_(self.read.weight, std=1 / hidden_size**0.5)
            nn.init.zeros_(self.read.bias)
            nn.init.normal_(self.write.weight, std=0.02)
        self.retention_logit = nn.Parameter(torch.tensor(-2.0))
        # Fixed (non-trained) statistics that center the read input. Without them every
        # activation shares one large common component, tanh saturates, and z stops
        # depending on the input. See fit_input_stats().
        self.register_buffer("input_mean", torch.zeros(hidden_size))
        self.register_buffer("input_scale", torch.ones(hidden_size))

    @torch.no_grad()
    def fit_input_stats(self, hidden: torch.Tensor) -> None:
        """Center and scale the read input using example activations [batch, hidden]."""
        if hidden.ndim != 2 or hidden.shape[0] < 2 or hidden.shape[1] != self.hidden_size:
            raise ValueError("expected at least two activations of shape [batch, hidden_size]")
        normed = F.layer_norm(hidden.to(self.read.weight), (self.hidden_size,))
        self.input_mean.copy_(normed.mean(dim=0))
        self.input_scale.copy_(normed.std(dim=0).clamp_min(1e-3))

    @property
    def beta(self) -> float:
        return float(self.retention_logit.detach().sigmoid())

    def read_state(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}")
        normed = F.layer_norm(hidden.to(self.read.weight), (self.hidden_size,))
        observed = torch.tanh(self.read((normed - self.input_mean) / self.input_scale))
        return observed.squeeze(-1) if self.latent_size == 1 else observed

    def update_state(self, previous: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        beta = self.retention_logit.sigmoid()
        return beta * previous + (1 - beta) * observed

    def write_state(self, hidden: torch.Tensor, z, alpha: float = 1.0) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}")
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite")
        state = torch.as_tensor(z, device=self.write.weight.device, dtype=self.write.weight.dtype)
        if self.latent_size == 1:
            state = state.unsqueeze(-1)
        if state.shape[-1] != self.latent_size:
            raise ValueError(f"expected latent size {self.latent_size}")
        delta = self.write(state).to(hidden)
        # Batched z [batch, k] broadcasts over a sequence, not over batch.
        while delta.ndim < hidden.ndim:
            delta = delta.unsqueeze(-2)
        return hidden + alpha * delta


def save_checkpoint(path: Path, relay: TrainableRelay, *, model_name: str,
                    read_layer: int, write_layer: int, alpha: float) -> None:
    """Save only relay parameters and explicit model/config metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 2,
        "model_name": model_name,
        "hidden_size": relay.hidden_size,
        "latent_size": relay.latent_size,
        "read_layer": read_layer,
        "write_layer": write_layer,
        "alpha": alpha,
        "state_dict": {name: value.detach().cpu() for name, value in relay.state_dict().items()},
    }, path)


def load_checkpoint(path: Path) -> tuple[TrainableRelay, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 2:
        raise ValueError("unsupported relay checkpoint version")
    relay = TrainableRelay(payload["hidden_size"], payload["latent_size"])
    relay.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in relay.state_dict().values()):
        raise ValueError("checkpoint contains non-finite parameters")
    relay.eval()
    return relay, {key: value for key, value in payload.items() if key != "state_dict"}

"""Metrics for measuring latent intervention effects."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LogitEffect:
    l2: float
    max_abs: float
    kl_divergence: float


def compare_logits(reference: torch.Tensor, changed: torch.Tensor) -> LogitEffect:
    """Measure a changed distribution against reference logits."""
    if reference.shape != changed.shape:
        raise ValueError("logit shapes must match")
    delta = changed - reference
    changed_log_probs = F.log_softmax(changed, dim=-1)
    changed_probs = changed_log_probs.exp()
    reference_log_probs = F.log_softmax(reference, dim=-1)
    kl = torch.sum(changed_probs * (changed_log_probs - reference_log_probs))
    return LogitEffect(delta.norm().item(), delta.abs().max().item(), kl.item())

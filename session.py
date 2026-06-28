"""A stateful product-facing interface for the latent relay."""

from dataclasses import dataclass
import math

import torch

from model import FrozenTransformer
from relay import LatentReader, LatentState, LatentWriter


@dataclass(frozen=True)
class RelayObservation:
    """The raw and stored values produced when remembering an interaction."""

    observed: float | torch.Tensor
    stored: float | torch.Tensor


class LatentRelay:
    """Carry detached scalar/vector state between independent transformer calls."""

    def __init__(
        self,
        transformer: FrozenTransformer,
        read_layer: int,
        write_layer: int,
        *,
        reader: LatentReader | None = None,
        writer: LatentWriter | None = None,
        beta: float = 0.0,
        alpha: float = 1.0,
    ) -> None:
        if not 0 <= read_layer <= transformer.layer_count:
            raise ValueError(f"read_layer must be in 0..{transformer.layer_count}")
        if not 1 <= write_layer <= transformer.layer_count:
            raise ValueError(f"write_layer must be in 1..{transformer.layer_count}")
        self.transformer = transformer
        self.read_layer = read_layer
        self.write_layer = write_layer
        self.reader = reader or LatentReader(transformer.hidden_size)
        self.writer = writer or LatentWriter(transformer.hidden_size)
        for component in (self.reader, self.writer):
            size = getattr(component, "hidden_size", None)
            if size is None:
                size = component.direction.numel()
            if size != transformer.hidden_size:
                raise ValueError("reader/writer hidden size must match transformer")
        latent_size = getattr(self.reader, "latent_size", 1)
        if latent_size != getattr(self.writer, "latent_size", 1):
            raise ValueError("reader/writer latent sizes must match")
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite")
        self.state = LatentState(beta, latent_size)
        self.alpha = alpha

    @classmethod
    def from_checkpoint(cls, transformer: FrozenTransformer, path) -> "LatentRelay":
        from trainable import load_checkpoint

        relay, metadata = load_checkpoint(path)
        if metadata["model_name"] != transformer.model_name:
            raise ValueError("checkpoint model does not match transformer")
        return cls(
            transformer, metadata["read_layer"], metadata["write_layer"],
            reader=relay, writer=relay, beta=relay.beta, alpha=metadata["alpha"],
        )

    @property
    def z(self) -> float | torch.Tensor:
        return self.state.z

    @torch.no_grad()
    def remember(self, prompt: str) -> RelayObservation:
        """Read a prompt and update the persistent state."""
        hidden = self.transformer.trace(prompt, [self.read_layer]).last_token_by_layer[
            self.read_layer
        ]
        observed = self.reader.read_state(hidden)
        stored = self.state.update(observed)
        observation = float(observed) if observed.ndim == 0 else observed.detach().cpu().clone()
        stored = stored.clone() if isinstance(stored, torch.Tensor) else stored
        return RelayObservation(observation, stored)

    def next_token_logits(self, prompt: str) -> torch.Tensor:
        """Predict with the current state written into the target prompt."""
        return self.transformer.next_token_logits(
            prompt,
            self.write_layer,
            lambda hidden: self.writer.write_state(hidden, self.z, self.alpha),
        )

    def generate(
        self, prompt: str, max_new_tokens: int = 32, temperature: float = 0.0
    ) -> str:
        """Generate text while carrying the same state through every token step."""
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        ids = self.transformer.encode(prompt)
        limit = self.transformer.model.config.max_position_embeddings
        if ids.shape[1] + max_new_tokens > limit:
            raise ValueError("generation exceeds the model context window")
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = self.transformer.differentiable_logits(
                    ids, self.write_layer,
                    lambda hidden: self.writer.write_state(hidden, self.z, self.alpha),
                )
                if temperature == 0:
                    token = logits.argmax().reshape(1, 1)
                else:
                    token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).reshape(1, 1)
                ids = torch.cat((ids, token.to(ids.device)), dim=-1)
                if int(token.item()) == self.transformer.tokenizer.eos_token_id:
                    break
        return self.transformer.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)

    def reset(self) -> None:
        """Clear the cross-interaction state."""
        self.state.reset()

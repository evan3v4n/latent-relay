"""A stateful product-facing interface for the latent relay."""

from dataclasses import dataclass

import torch

from model import FrozenTransformer
from relay import LatentReader, LatentState, LatentWriter


@dataclass(frozen=True)
class RelayObservation:
    """The raw and stored values produced when remembering an interaction."""

    observed: float
    stored: float


class LatentRelay:
    """Carry one scalar latent value between independent transformer calls."""

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
        self.state = LatentState(beta)
        self.alpha = alpha

    @property
    def z(self) -> float:
        return self.state.z

    def remember(self, prompt: str) -> RelayObservation:
        """Read a prompt and update the persistent state."""
        hidden = self.transformer.trace(prompt, [self.read_layer]).last_token_by_layer[
            self.read_layer
        ]
        observed = self.reader.read_state(hidden)
        stored = self.state.update(observed)
        return RelayObservation(float(observed), stored)

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
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        text = prompt
        for _ in range(max_new_tokens):
            logits = self.next_token_logits(text)
            if temperature == 0:
                token_id = int(torch.argmax(logits))
            else:
                token_id = int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1))
            text += self.transformer.tokenizer.decode([token_id])
        return text

    def reset(self) -> None:
        """Clear the cross-interaction state."""
        self.state.reset()

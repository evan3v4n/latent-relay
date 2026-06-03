"""Frozen transformer access and activation extraction."""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "EleutherAI/pythia-160m"


@dataclass(frozen=True)
class ActivationTrace:
    prompt: str
    tokens: tuple[str, ...]
    # Hugging Face index 0 is the embedding output; index n is block n's output.
    last_token_by_layer: dict[int, torch.Tensor]


class FrozenTransformer:
    """Small, reusable interface for later relay read/write experiments."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def layer_count(self) -> int:
        return self.model.config.num_hidden_layers

    @torch.inference_mode()
    def trace(self, prompt: str, layers: list[int] | None = None) -> ActivationTrace:
        """Return a detached [hidden_size] vector for the prompt's final token."""
        if not prompt.strip():
            raise ValueError("prompt must contain text")
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model(**encoded, output_hidden_states=True, return_dict=True)
        hidden_states = output.hidden_states
        selected = layers if layers is not None else list(range(len(hidden_states)))
        invalid = [layer for layer in selected if not 0 <= layer < len(hidden_states)]
        if invalid:
            raise ValueError(f"layers {invalid} outside valid range 0..{len(hidden_states) - 1}")

        token_ids = encoded["input_ids"][0].tolist()
        activations = {
            layer: hidden_states[layer][0, -1].detach().float().cpu()
            for layer in selected
        }
        return ActivationTrace(
            prompt=prompt,
            tokens=tuple(self.tokenizer.convert_ids_to_tokens(token_ids)),
            last_token_by_layer=activations,
        )

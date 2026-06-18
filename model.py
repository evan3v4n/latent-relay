"""Frozen transformer access and activation extraction."""

from collections.abc import Callable
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
            last_layer = len(hidden_states) - 1
            raise ValueError(f"layers {invalid} outside valid range 0..{last_layer}")

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

    @torch.inference_mode()
    def next_token_logits(
        self,
        prompt: str,
        write_layer: int | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return next-token logits, optionally transforming one block's output."""
        if not prompt.strip():
            raise ValueError("prompt must contain text")
        if (write_layer is None) != (transform is None):
            raise ValueError("write_layer and transform must be provided together")
        if write_layer is not None and not 1 <= write_layer <= self.layer_count:
            raise ValueError(f"write_layer must be in 1..{self.layer_count}")

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        handle = None
        if transform is not None:
            block = self.model.gpt_neox.layers[write_layer - 1]

            def hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    return (transform(output[0]), *output[1:])
                return transform(output)

            handle = block.register_forward_hook(hook)

        try:
            output = self.model(**encoded, return_dict=True)
        finally:
            if handle is not None:
                handle.remove()
        return output.logits[0, -1].detach().float().cpu()

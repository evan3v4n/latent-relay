"""Frozen Pythia access, activation extraction, and scoped interventions."""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "EleutherAI/pythia-160m"


@dataclass(frozen=True)
class ActivationTrace:
    prompt: str
    tokens: tuple[str, ...]
    # HF index 0 is embeddings; the final index includes final layer normalization.
    last_token_by_layer: dict[int, torch.Tensor]


class FrozenTransformer:
    """Single-prompt Pythia interface. A shared instance requires serialized calls."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Explicit fp32 avoids model-config-dependent quantization of small effects.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(self.device)
        if self.model.config.model_type != "gpt_neox":
            raise ValueError("layer interventions currently require a GPT-NeoX/Pythia model")
        self.model.eval()
        self.model.requires_grad_(False)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def layer_count(self) -> int:
        return self.model.config.num_hidden_layers

    def encode(self, prompt: str) -> torch.Tensor:
        if not prompt.strip():
            raise ValueError("prompt must contain text")
        ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        self._validate_tokens(ids)
        return ids

    def _validate_tokens(self, ids: torch.Tensor) -> None:
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [1, non-empty sequence]")
        if ids.shape[1] > self.model.config.max_position_embeddings:
            raise ValueError("prompt exceeds the model context window")

    @torch.inference_mode()
    def trace(self, prompt: str, layers: list[int] | None = None) -> ActivationTrace:
        selected = layers if layers is not None else list(range(self.layer_count + 1))
        invalid = [layer for layer in selected if not 0 <= layer <= self.layer_count]
        if invalid:
            raise ValueError(f"layers {invalid} outside valid range 0..{self.layer_count}")
        encoded = self.encode(prompt)
        output = self.model(
            input_ids=encoded, output_hidden_states=True, return_dict=True, use_cache=False
        )
        return ActivationTrace(
            prompt=prompt,
            tokens=tuple(self.tokenizer.convert_ids_to_tokens(encoded[0].tolist())),
            last_token_by_layer={
                layer: output.hidden_states[layer][0, -1].detach().float().cpu()
                for layer in selected
            },
        )

    @contextmanager
    def intervention(self, write_layer=None, transform=None):
        """Remove the temporary hook even when a transform/forward fails."""
        if (write_layer is None) != (transform is None):
            raise ValueError("write_layer and transform must be provided together")
        if write_layer is not None and not 1 <= write_layer <= self.layer_count:
            raise ValueError(f"write_layer must be in 1..{self.layer_count}")
        handle = None
        if transform is not None:
            def hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    return (transform(output[0]), *output[1:])
                return transform(output)

            handle = self.model.gpt_neox.layers[write_layer - 1].register_forward_hook(hook)
        try:
            yield
        finally:
            if handle is not None:
                handle.remove()

    def differentiable_logits(
        self,
        input_ids: torch.Tensor,
        write_layer: int | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Retain intervention gradients while every base-model weight stays frozen."""
        self._validate_tokens(input_ids)
        with self.intervention(write_layer, transform):
            output = self.model(
                input_ids=input_ids.to(self.device), return_dict=True, use_cache=False
            )
        return output.logits[0, -1].float()

    @torch.inference_mode()
    def next_token_logits(
        self,
        prompt: str,
        write_layer: int | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.differentiable_logits(self.encode(prompt), write_layer, transform).cpu()

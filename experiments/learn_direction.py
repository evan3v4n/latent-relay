"""Phase 7: fit a semantic direction from labeled prompt activations."""

import argparse

import torch

from model import DEFAULT_MODEL, FrozenTransformer
from relay import LatentReader, fit_direction


DEFAULT_POSITIVE = ("I am happy today.", "That was a wonderful surprise.")
DEFAULT_NEGATIVE = ("I am sad today.", "That was a terrible surprise.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive", nargs="+", default=DEFAULT_POSITIVE)
    parser.add_argument("--negative", nargs="+", default=DEFAULT_NEGATIVE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--layer", type=int, help="hidden-state index (default: final block)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transformer = FrozenTransformer(args.model, args.device)
    layer = transformer.layer_count if args.layer is None else args.layer

    def activations(prompts: list[str]) -> torch.Tensor:
        return torch.stack(
            [transformer.trace(prompt, [layer]).last_token_by_layer[layer] for prompt in prompts]
        )

    direction = fit_direction(activations(args.positive), activations(args.negative))
    reader = LatentReader.from_direction(direction)
    print(f"layer: {layer}")
    print("positive examples:")
    for prompt in args.positive:
        value = reader.read_state(transformer.trace(prompt, [layer]).last_token_by_layer[layer])
        print(f"  {value.item():+.4f}  {prompt}")
    print("negative examples:")
    for prompt in args.negative:
        value = reader.read_state(transformer.trace(prompt, [layer]).last_token_by_layer[layer])
        print(f"  {value.item():+.4f}  {prompt}")


if __name__ == "__main__":
    main()

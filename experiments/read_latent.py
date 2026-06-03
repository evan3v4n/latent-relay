"""Phase 2: read a scalar latent value from final-token activations."""

import argparse

from model import DEFAULT_MODEL, FrozenTransformer
from relay import LatentReader

DEFAULT_PROMPTS = (
    "I am extremely happy today.",
    "I just lost something important.",
    "The number seven is prime.",
    "Paris is in France.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="*", help="prompts to measure")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument(
        "--layer", type=int, help="hidden-state index (default: final block)"
    )
    parser.add_argument("--seed", type=int, default=0, help="reader direction seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    print(f"loading {args.model} on {args.device}...")
    transformer = FrozenTransformer(args.model, args.device)
    layer = transformer.layer_count if args.layer is None else args.layer
    reader = LatentReader(transformer.hidden_size, args.seed)
    print(f"reading layer {layer} with direction seed {args.seed}\n")

    print(" latent  prompt")
    print("-------  ------")
    for prompt in prompts:
        hidden = transformer.trace(prompt, [layer]).last_token_by_layer[layer]
        z = reader.read_state(hidden).item()
        print(f"{z:>7.4f}  {prompt}")


if __name__ == "__main__":
    main()

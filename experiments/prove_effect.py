"""Phase 5: measure the causal effect of manually changing latent state."""

import argparse

import torch

from metrics import compare_logits
from model import DEFAULT_MODEL, FrozenTransformer
from relay import LatentWriter

DEFAULT_PROMPT = "The person opened the door and saw"
STATES = (-1.0, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--write-layer", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--write-seed", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    return args


def top_predictions(
    transformer: FrozenTransformer, logits: torch.Tensor, count: int
) -> str:
    probabilities = torch.softmax(logits, dim=-1)
    values, token_ids = probabilities.topk(min(count, logits.numel()))
    return ", ".join(
        f"{transformer.tokenizer.decode([token_id])!r} {probability:.3%}"
        for probability, token_id in zip(values.tolist(), token_ids.tolist())
    )


def main() -> None:
    args = parse_args()
    print(f"loading {args.model} on {args.device}...")
    transformer = FrozenTransformer(args.model, args.device)
    writer = LatentWriter(transformer.hidden_size, args.write_seed)

    logits = {
        z: transformer.next_token_logits(
            args.prompt,
            args.write_layer,
            lambda hidden, state=z: writer.write_state(hidden, state, args.alpha),
        )
        for z in STATES
    }
    reference = logits[0.0]

    print(f"prompt: {args.prompt!r}")
    print(f"write:  layer {args.write_layer}, alpha={args.alpha:g}\n")
    print("   z    logit L2    max |Δ|          KL  top predictions")
    print("----  ----------  ---------  ----------  ---------------")
    for z in STATES:
        effect = compare_logits(reference, logits[z])
        predictions = top_predictions(transformer, logits[z], args.top_k)
        print(
            f"{z:>+4.0f}  {effect.l2:>10.4f}  {effect.max_abs:>9.4f}  "
            f"{effect.kl_divergence:>10.6f}  {predictions}"
        )


if __name__ == "__main__":
    main()

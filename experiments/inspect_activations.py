"""Phase 1: inspect and compare final-token hidden activations."""

import argparse

import torch.nn.functional as F

from model import DEFAULT_MODEL, ActivationTrace, FrozenTransformer

DEFAULT_PROMPTS = (
    "The dog ran across the",
    "The stock market crashed because",
)


def describe(trace: ActivationTrace) -> None:
    print(f"\nprompt: {trace.prompt!r}")
    print(f"tokens ({len(trace.tokens)}): {' | '.join(trace.tokens)}")


def compare(first: ActivationTrace, second: ActivationTrace) -> None:
    print("\nlayer      shape     first norm  second norm  cosine")
    print("-----  ---------  ------------  -----------  -------")
    for layer, left in first.last_token_by_layer.items():
        right = second.last_token_by_layer[layer]
        cosine = F.cosine_similarity(left, right, dim=0).item()
        print(
            f"{layer:>5}  {str(list(left.shape)):>9}  "
            f"{left.norm().item():>12.4f}  {right.norm().item():>11.4f}  {cosine:>7.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", nargs="*", help="two prompts to compare")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--layers", type=int, nargs="+", help="hidden-state indices")
    args = parser.parse_args()
    if args.prompts and len(args.prompts) != 2:
        parser.error("provide exactly two prompts, or none to use the defaults")
    return args


def main() -> None:
    args = parse_args()
    prompts = tuple(args.prompts) or DEFAULT_PROMPTS
    print(f"loading {args.model} on {args.device}...")
    probe = FrozenTransformer(args.model, args.device)
    print(
        f"ready: {probe.layer_count} transformer blocks, "
        f"hidden size {probe.hidden_size} (layer 0 is embeddings)"
    )

    traces = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"forward pass {index}/{len(prompts)}...")
        traces.append(probe.trace(prompt, args.layers))
    for trace in traces:
        describe(trace)
    compare(*traces)


if __name__ == "__main__":
    main()

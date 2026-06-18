"""Phase 4: carry a scalar state from one prompt into another."""

import argparse

import torch

from model import DEFAULT_MODEL, FrozenTransformer
from relay import LatentReader, LatentState, LatentWriter

DEFAULT_SOURCE = "I am extremely happy today."
DEFAULT_TARGET = "The person opened the door and saw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--read-layer", type=int, default=12)
    parser.add_argument("--write-layer", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--read-seed", type=int, default=0)
    parser.add_argument("--write-seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"loading {args.model} on {args.device}...")
    transformer = FrozenTransformer(args.model, args.device)
    reader = LatentReader(transformer.hidden_size, args.read_seed)
    writer = LatentWriter(transformer.hidden_size, args.write_seed)
    state = LatentState(args.beta)

    hidden = transformer.trace(args.source, [args.read_layer]).last_token_by_layer[
        args.read_layer
    ]
    observed = reader.read_state(hidden)
    state.update(observed)

    baseline = transformer.next_token_logits(args.target)
    injected = transformer.next_token_logits(
        args.target,
        args.write_layer,
        lambda activation: writer.write_state(activation, state.z, args.alpha),
    )
    delta = injected - baseline

    print(f"source:      {args.source!r}")
    print(f"target:      {args.target!r}")
    print(f"observed z:  {observed.item():.6f}")
    print(f"stored z:    {state.z:.6f}  (beta={state.beta:g})")
    print(
        f"write:       layer {args.write_layer}, alpha={args.alpha:g}\n"
        f"logit delta: L2={delta.norm().item():.6f}, max={delta.abs().max().item():.6f}"
    )
    token_id = torch.argmax(injected).item()
    print(f"top token:   {transformer.tokenizer.decode([token_id])!r}")


if __name__ == "__main__":
    main()

"""Phase 6: compare latent effects across read/write-layer pairs."""

import argparse
import csv
from pathlib import Path

from metrics import LogitEffect, compare_logits
from model import DEFAULT_MODEL, FrozenTransformer
from relay import LatentReader, LatentState, LatentWriter

DEFAULT_SOURCE = "I am extremely happy today."
DEFAULT_TARGET = "The person opened the door and saw"
DEFAULT_PAIRS = ((4, 4), (4, 8), (8, 8), (8, 10))


def layer_pair(value: str) -> tuple[int, int]:
    try:
        read, write = map(int, value.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use READ:WRITE, for example 4:8") from error
    return read, write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--pairs", nargs="+", type=layer_pair, default=DEFAULT_PAIRS)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--read-seed", type=int, default=0)
    parser.add_argument("--write-seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/phase6.csv"))
    return parser.parse_args()


def write_results(
    path: Path, rows: list[tuple[int, int, float, float, LogitEffect]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ("read_layer", "write_layer", "alpha", "z", "l2", "max_abs", "kl")
        )
        for read, write, alpha, z, effect in rows:
            writer.writerow(
                (read, write, alpha, z, effect.l2, effect.max_abs, effect.kl_divergence)
            )


def main() -> None:
    args = parse_args()
    print(f"loading {args.model} on {args.device}...")
    transformer = FrozenTransformer(args.model, args.device)
    reader = LatentReader(transformer.hidden_size, args.read_seed)
    writer = LatentWriter(transformer.hidden_size, args.write_seed)
    state = LatentState()

    read_layers = sorted({read for read, _ in args.pairs})
    trace = transformer.trace(args.source, read_layers)
    baseline = transformer.next_token_logits(args.target)
    rows = []
    for read_layer, write_layer in args.pairs:
        state.reset()
        z = state.update(reader.read_state(trace.last_token_by_layer[read_layer]))
        changed = transformer.next_token_logits(
            args.target,
            write_layer,
            lambda hidden, value=z: writer.write_state(hidden, value, args.alpha),
        )
        rows.append(
            (read_layer, write_layer, args.alpha, z, compare_logits(baseline, changed))
        )

    print(f"source: {args.source!r}\ntarget: {args.target!r}\n")
    print(" read  write  alpha         z    logit L2    max |Δ|          KL")
    print("-----  -----  -----  --------  ----------  ---------  ----------")
    for read, write, alpha, z, effect in rows:
        print(
            f"{read:>5}  {write:>5}  {alpha:>5g}  {z:>8.4f}  "
            f"{effect.l2:>10.4f}  {effect.max_abs:>9.4f}  "
            f"{effect.kl_divergence:>10.6f}"
        )
    write_results(args.output, rows)
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()

"""Phase 7: does a fitted direction read sentiment on sentences it never saw?

For every layer we repeatedly split the explicit sentences into a "fit" half and a
"held-out" half, fit a direction on the fit half, and score the held-out half. Controls
(random direction, shuffled labels) show what chance looks like. A separate set of
implicit-sentiment sentences is never used for choosing a layer, so it is a fair check
that the direction is not just keying on a few emotion words.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import DEFAULT_MODEL, FrozenTransformer
from relay import CalibratedReader
import sentiment_data as data
from stats import auroc

SPLITS = 5
RANDOM_CONTROLS = 20


def short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def collect_activations(transformer: FrozenTransformer, sentences) -> dict[int, torch.Tensor]:
    """Last-token activation of every sentence at every layer: {layer: [n, hidden]}."""
    traces = [transformer.trace(sentence).last_token_by_layer for sentence in sentences]
    return {layer: torch.stack([trace[layer] for trace in traces])
            for layer in range(transformer.layer_count + 1)}


def make_split(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(count)
    return order[: count // 2], order[count // 2:]


def project(hidden: torch.Tensor, direction: torch.Tensor, center: torch.Tensor) -> np.ndarray:
    return ((hidden - center) @ direction).numpy()


def score_split(pos, neg, imp_pos, imp_neg, split_seed):
    """All Phase 7 measurements for one layer and one random split."""
    pos_fit, pos_held = make_split(len(pos), split_seed)
    neg_fit, neg_held = make_split(len(neg), split_seed + 1000)
    reader = CalibratedReader.from_examples(pos[pos_fit], neg[neg_fit])

    def evaluate(reader, positives, negatives):
        z = reader.read_state(torch.cat((positives, negatives))).numpy()
        labels = np.array([True] * len(positives) + [False] * len(negatives))
        return float(((z > 0) == labels).mean()), auroc(z, labels)

    accuracy, area = evaluate(reader, pos[pos_held], neg[neg_held])
    implicit_accuracy, implicit_area = evaluate(reader, imp_pos, imp_neg)

    # Control 1: labels shuffled inside the fit half -> direction carries no sentiment.
    rng = np.random.default_rng(split_seed + 2000)
    fit_pool = torch.cat((pos[pos_fit], neg[neg_fit]))
    mixed = rng.permutation(len(fit_pool))
    shuffled = CalibratedReader.from_examples(fit_pool[mixed[: len(pos_fit)]], fit_pool[mixed[len(pos_fit):]])
    shuffled_accuracy, shuffled_area = evaluate(shuffled, pos[pos_held], neg[neg_held])

    # Control 2: random unit directions, same centering as the real reader.
    random_accuracy, random_area = [], []
    held = torch.cat((pos[pos_held], neg[neg_held]))
    labels = np.array([True] * len(pos_held) + [False] * len(neg_held))
    for _ in range(RANDOM_CONTROLS):
        direction = torch.nn.functional.normalize(
            torch.from_numpy(rng.standard_normal(held.shape[1])).float(), dim=0)
        values = project(held, direction, reader.center)
        random_accuracy.append(float(((values > 0) == labels).mean()))
        random_area.append(auroc(values, labels))
    return {
        "accuracy": accuracy, "auroc": area,
        "implicit_accuracy": implicit_accuracy, "implicit_auroc": implicit_area,
        "shuffled_accuracy": shuffled_accuracy, "shuffled_auroc": shuffled_area,
        "random_accuracy": float(np.mean(random_accuracy)), "random_auroc": float(np.mean(random_area)),
    }


def sweep_layers(pos_acts, neg_acts, imp_pos_acts, imp_neg_acts, layers):
    rows = []
    for layer in layers:
        per_split = [score_split(pos_acts[layer], neg_acts[layer], imp_pos_acts[layer],
                                 imp_neg_acts[layer], seed) for seed in range(SPLITS)]
        row = {"layer": layer}
        for key in per_split[0]:
            values = [item[key] for item in per_split]
            row[key] = float(np.mean(values))
            row[key + "_std"] = float(np.std(values))
        rows.append(row)
    return rows


def select_read_layer(rows) -> int:
    """Best layer by held-out explicit AUROC. Implicit data is deliberately not consulted."""
    best = max(rows, key=lambda row: (round(row["auroc"], 3), round(row["accuracy"], 3), -row["layer"]))
    return best["layer"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="default: results/phase7_<model>.json")
    return parser.parse_args()


def main():
    args = parse_args()
    transformer = FrozenTransformer(args.model, args.device)
    name = short_name(args.model)
    output = args.output or Path(f"results/phase7_{name}.json")
    pos = collect_activations(transformer, data.EXPLICIT_POSITIVE)
    neg = collect_activations(transformer, data.EXPLICIT_NEGATIVE)
    imp_pos = collect_activations(transformer, data.IMPLICIT_POSITIVE)
    imp_neg = collect_activations(transformer, data.IMPLICIT_NEGATIVE)
    layers = range(1, transformer.layer_count + 1)
    rows = sweep_layers(pos, neg, imp_pos, imp_neg, layers)
    best = select_read_layer(rows)

    print(f"{'layer':>5} {'acc':>6} {'auroc':>6} {'implicit':>9} {'shuffled':>9} {'random':>7}")
    for row in rows:
        mark = "  <- chosen" if row["layer"] == best else ""
        print(f"{row['layer']:>5} {row['accuracy']:>6.1%} {row['auroc']:>6.3f} "
              f"{row['implicit_accuracy']:>9.1%} {row['shuffled_accuracy']:>9.1%} "
              f"{row['random_accuracy']:>7.1%}{mark}")

    chosen = next(row for row in rows if row["layer"] == best)
    report = {
        "model": args.model, "splits": SPLITS, "random_controls": RANDOM_CONTROLS,
        "examples": {"explicit_per_class": len(data.EXPLICIT_POSITIVE),
                     "implicit_per_class": len(data.IMPLICIT_POSITIVE)},
        "chosen_read_layer": best,
        "criteria": {
            "held_out_accuracy_at_least_85": chosen["accuracy"] >= 0.85,
            "random_control_near_chance": 0.4 <= chosen["random_accuracy"] <= 0.6,
            "shuffled_control_near_chance": 0.4 <= chosen["shuffled_accuracy"] <= 0.6,
            "implicit_accuracy_at_least_75": chosen["implicit_accuracy"] >= 0.75,
        },
        "layers": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"chosen read layer: {best}")
    for criterion, passed in report["criteria"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {criterion}")
    print(f"saved {output}")


if __name__ == "__main__":
    main()

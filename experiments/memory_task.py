"""Train and evaluate a relay on a balanced, separated-context secret-code task."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import random

import torch
import torch.nn.functional as F
import transformers

from metrics import compare_logits
from model import DEFAULT_MODEL, FrozenTransformer
from trainable import TrainableRelay, load_checkpoint, save_checkpoint

LABELS = ("A", "B", "C", "D")
TRAIN_SOURCES = (
    "Remember this secret code: {label}",
    "Store the following code for later: {label}",
    "The private code to remember is {label}",
    "Keep this code in memory: {label}",
)
TEST_SOURCES = (
    "For the next question, retain this code: {label}",
    "My confidential code is {label}",
    "Please memorize the code {label}",
    "You will need this code later: {label}",
)
TRAIN_TARGETS = ("The secret code is", "Recall the stored code. Answer:")
TEST_TARGETS = ("What was the private code? Answer:", "The code you remember is")


def examples(held_out=False):
    sources = TEST_SOURCES if held_out else TRAIN_SOURCES
    targets = TEST_TARGETS if held_out else TRAIN_TARGETS
    return [
        {"source": template.format(label=label), "target": targets[index % len(targets)], "label": label}
        for index, template in enumerate(sources) for label in LABELS
    ]


def candidate_ids(transformer):
    ids = [transformer.tokenizer.encode(" " + label, add_special_tokens=False) for label in LABELS]
    if any(len(tokens) != 1 for tokens in ids):
        raise ValueError("this task requires each space-prefixed code to be one token")
    flattened = [tokens[0] for tokens in ids]
    if len(set(flattened)) != len(LABELS):
        raise ValueError("code tokens must be distinct")
    return flattened


def hidden_examples(transformer, records, read_layer):
    # Stack outside inference_mode: training must be able to save these for backward.
    return torch.stack([
        transformer.trace(item["source"], [read_layer]).last_token_by_layer[read_layer]
        for item in records
    ]).clone()


def train(transformer, relay, *, read_layer, write_layer, alpha, epochs, lr, seed):
    records = examples()
    activations = hidden_examples(transformer, records, read_layer)
    relay.fit_input_stats(activations)
    targets = {item["target"]: transformer.encode(item["target"]) for item in records}
    ids = candidate_ids(transformer)
    optimizer = torch.optim.Adam(relay.parameters(), lr=lr)
    generator = random.Random(seed)
    history = []
    relay.train()
    for epoch in range(epochs):
        order = list(range(len(records)))
        generator.shuffle(order)
        total = 0.0
        for index in order:
            item = records[index]
            observed = relay.read_state(activations[index])
            z = relay.update_state(torch.zeros_like(observed), observed)
            logits = transformer.differentiable_logits(
                targets[item["target"]], write_layer,
                lambda hidden: relay.write_state(hidden, z, alpha),
            )
            # Full-vocabulary cross entropy: candidate-only success is reported separately.
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([ids[LABELS.index(item["label"])]], device=logits.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(relay.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        history.append(total / len(records))
        with torch.no_grad():
            saturated = (relay.read_state(activations).abs() > 0.99).float().mean().item()
        print(f"epoch {epoch + 1}/{epochs}: loss={history[-1]:.4f} saturated={saturated:.0%}", flush=True)
    relay.eval()
    return history


@torch.no_grad()
def evaluate(transformer, relay, *, read_layer, write_layer, alpha, seed):
    records = examples(held_out=True)
    activations = hidden_examples(transformer, records, read_layer)
    ids = candidate_ids(transformer)
    raw = relay.read_state(activations)
    states = relay.update_state(torch.zeros_like(raw), raw)
    initial = TrainableRelay(relay.hidden_size, relay.latent_size, seed).to(transformer.device)
    initial.input_mean.copy_(relay.input_mean)
    initial.input_scale.copy_(relay.input_scale)
    untrained_raw = initial.read_state(activations)
    untrained_states = initial.update_state(torch.zeros_like(untrained_raw), untrained_raw)
    rows = []
    baseline_cache = {}
    distractor = transformer.trace("The number seven is prime.", [read_layer]).last_token_by_layer[read_layer]
    distractor_state = relay.read_state(distractor)
    for index, item in enumerate(records):
        target = item["target"]
        if target not in baseline_cache:
            baseline_cache[target] = transformer.next_token_logits(target)
        baseline = baseline_cache[target]
        # A cyclic label shift within each balanced source group destroys the correct association.
        wrong_index = (index // len(LABELS)) * len(LABELS) + (index + 1) % len(LABELS)
        drifted = states[index]
        drift_states = {}
        for step in range(1, 4):
            drifted = relay.update_state(drifted, distractor_state)
            if step in (1, 3):
                drift_states[f"relay_after_{step}_distractors"] = drifted
        conditions = {
            "relay": (relay, states[index]),
            "zero_state": (relay, torch.zeros_like(states[index])),
            "reset": (relay, torch.zeros_like(states[index])),
            "shuffled_state": (relay, states[wrong_index]),
            "untrained": (initial, untrained_states[index]),
            **{name: (relay, state) for name, state in drift_states.items()},
        }
        for condition in (*conditions, "text_only"):
            if condition == "text_only":
                logits = transformer.next_token_logits(item["source"] + "\n" + target)
                state = torch.zeros_like(states[index])
            else:
                module, state = conditions[condition]
                logits = transformer.next_token_logits(
                    target, write_layer, lambda hidden: module.write_state(hidden, state, alpha)
                )
            truth = ids[LABELS.index(item["label"])]
            candidate_prediction = LABELS[int(logits[ids].argmax())]
            effect = compare_logits(baseline, logits)
            rows.append({
                **item, "condition": condition,
                "candidate_prediction": candidate_prediction,
                "candidate_correct": candidate_prediction == item["label"],
                "full_vocab_correct": int(logits.argmax()) == truth,
                "nll": float(-logits.log_softmax(-1)[truth]),
                "state_norm": float(state.norm()),
                "kl_from_zero": effect.kl_divergence,
            })
        # Repeated independent queries do not update latent state.
        repeated = transformer.next_token_logits(target, write_layer, lambda h: relay.write_state(h, states[index], alpha))
        original = transformer.next_token_logits(target, write_layer, lambda h: relay.write_state(h, states[index], alpha))
        if not torch.equal(repeated, original):
            raise RuntimeError("persistent reads unexpectedly changed predictions")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    summary = {
        name: {
            "count": len(group),
            "candidate_accuracy": sum(row["candidate_correct"] for row in group) / len(group),
            "full_vocab_accuracy": sum(row["full_vocab_correct"] for row in group) / len(group),
            "mean_nll": sum(row["nll"] for row in group) / len(group),
            "mean_kl_from_zero": sum(row["kl_from_zero"] for row in group) / len(group),
        } for name, group in grouped.items()
    }
    # Direct test of the memory channel itself: can the label be read back from z alone?
    train_records = examples()
    train_states = relay.update_state(torch.zeros(len(train_records), relay.latent_size).to(states),
                                      relay.read_state(hidden_examples(transformer, train_records, read_layer)))
    centroids = torch.stack([
        train_states[[i for i, r in enumerate(train_records) if r["label"] == label]].mean(0)
        for label in LABELS])
    held_labels = torch.tensor([LABELS.index(r["label"]) for r in records])
    nearest = torch.cdist(states, centroids).argmin(dim=1)
    summary["state_readout"] = {
        "nearest_centroid_accuracy": (nearest == held_labels).float().mean().item(),
        "saturated_fraction": (states.abs() > 0.99).float().mean().item(),
    }
    return summary, rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--read-layer", type=int, default=8)
    parser.add_argument("--write-layer", type=int, default=12)
    parser.add_argument("--latent-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/relay.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/memory.json"))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.latent_size < 1 or args.threads < 1 or args.lr <= 0:
        parser.error("epochs, latent-size, threads, and lr must be positive")
    if not torch.isfinite(torch.tensor([args.alpha, args.lr])).all():
        parser.error("alpha and lr must be finite")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    transformer = FrozenTransformer(args.model, args.device)
    if args.evaluate_only:
        relay, config = load_checkpoint(args.checkpoint)
        if config["model_name"] != args.model or config["hidden_size"] != transformer.hidden_size:
            raise ValueError("checkpoint model does not match")
        args.read_layer, args.write_layer, args.alpha = config["read_layer"], config["write_layer"], config["alpha"]
        args.latent_size = relay.latent_size
        relay.to(transformer.device)
        history = []
    else:
        relay = TrainableRelay(transformer.hidden_size, args.latent_size, args.seed).to(transformer.device)
        history = train(transformer, relay, read_layer=args.read_layer, write_layer=args.write_layer,
                        alpha=args.alpha, epochs=args.epochs, lr=args.lr, seed=args.seed)
        save_checkpoint(args.checkpoint, relay, model_name=args.model, read_layer=args.read_layer,
                        write_layer=args.write_layer, alpha=args.alpha)
    summary, rows = evaluate(transformer, relay, read_layer=args.read_layer,
                             write_layer=args.write_layer, alpha=args.alpha, seed=args.seed)
    report = {
        "task": "Four-code recall across separated contexts; held-out source and target templates",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__},
        "train_examples": examples(), "train_loss": history,
        "trainable_parameters": sum(p.numel() for p in relay.parameters()),
        "learned_beta": relay.beta, "summary": summary, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"state_readout: {summary.pop('state_readout')}")
    for name, metrics in summary.items():
        print(f"{name}: candidate={metrics['candidate_accuracy']:.1%}, full_vocab={metrics['full_vocab_accuracy']:.1%}, nll={metrics['mean_nll']:.3f}")
    print(f"saved {args.output} and {args.checkpoint}")


if __name__ == "__main__":
    main()

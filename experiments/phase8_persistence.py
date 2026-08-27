"""Phase 8: can one stored number from an earlier interaction steer a later one?

Interaction 1 shows the model a sentence with a clear sentiment. We keep only the scalar
state z (the text is thrown away). Interaction 2 is a fresh, neutral prompt such as
"The movie turned out to be", with z written back into the model. We measure whether the
next-word distribution moves toward positive or negative words in the direction of z.

How the pieces are chosen (so the final numbers are not tuned on the test prompts):
  * The reader and writer directions are fitted on a "fit" half of the sentences.
  * Sentences from the other half provide the interaction-1 text, split into dev and test.
  * The write layer and strength are chosen on DEV prompts only (3 subjects).
  * All reported results use TEST prompts (9 other subjects).

Strength is measured in "class gaps": 1.0 means z=+1 moves the activation by the average
distance between positive and negative example sentences at the write layer.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.phase7_sentiment import (
    collect_activations, make_split, select_read_layer, short_name, sweep_layers,
)
from metrics import compare_logits
from model import DEFAULT_MODEL, FrozenTransformer
from relay import CalibratedReader, LatentWriter, fit_direction
from session import LatentRelay
import sentiment_data as data
from stats import bootstrap_ci, sign_flip_p, spearman

KL_BUDGET = 0.5           # mean nats between normal and steered next-word distributions
PERPLEXITY_BUDGET = 1.25  # steered perplexity / normal perplexity on neutral text
STRENGTHS = (0.5, 1.0, 2.0, 4.0, 8.0)
Z_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
STRENGTH_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
PERSISTENCE_BETAS = (0.5, 0.8)
PERSISTENCE_STEPS = (0, 1, 3, 5)
# Settings are picked with a safety margin below the reported budget: fluency measured on
# a small dev set was too optimistic about the held-out report set.
DEV_PERPLEXITY_BUDGET = 1.15
DEV_NEUTRAL = data.NEUTRAL_DEV         # used to pick the setting
REPORT_NEUTRAL = data.NEUTRAL_REPORT   # used only to report fluency
GENERATION_TOKENS = 20
GENERATION_SOURCES = 3


def word_ids(tokenizer, words):
    ids = []
    for word in words:
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(tokens) == 1:
            ids.append(tokens[0])
    return sorted(set(ids))


class Scorer:
    """Turns next-word logits into one number: log P(positive words) - log P(negative words)."""

    def __init__(self, tokenizer):
        self.positive = word_ids(tokenizer, data.POSITIVE_WORDS)
        self.negative = word_ids(tokenizer, data.NEGATIVE_WORDS)
        if len(self.positive) < 5 or len(self.negative) < 5:
            raise ValueError("too few single-token sentiment words for this tokenizer")

    def __call__(self, logits: torch.Tensor) -> float:
        log_probs = logits.log_softmax(-1)
        return float(log_probs[self.positive].logsumexp(0) - log_probs[self.negative].logsumexp(0))


def lexicon_score(text: str) -> int:
    words = text.lower().split()
    positive = sum(any(word.startswith(stem) for stem in data.LEXICON_POSITIVE) for word in words)
    negative = sum(any(word.startswith(stem) for stem in data.LEXICON_NEGATIVE) for word in words)
    return positive - negative


def random_direction(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(size, generator=generator), dim=0)


class Experiment:
    """Everything fitted once: reader, per-layer write directions, sources and targets."""

    def __init__(self, transformer: FrozenTransformer, read_layer: int):
        self.transformer = transformer
        self.read_layer = read_layer
        self.scorer = Scorer(transformer.tokenizer)
        self.pos = collect_activations(transformer, data.EXPLICIT_POSITIVE)
        self.neg = collect_activations(transformer, data.EXPLICIT_NEGATIVE)
        pos_fit, pos_held = make_split(len(data.EXPLICIT_POSITIVE), 0)
        neg_fit, neg_held = make_split(len(data.EXPLICIT_NEGATIVE), 1000)
        self.pos_fit, self.neg_fit = pos_fit, neg_fit
        self.neutral = collect_activations(transformer, data.NEUTRAL_CENTER)
        self.reader = CalibratedReader.from_examples(
            self.pos[read_layer][pos_fit], self.neg[read_layer][neg_fit], self.neutral[read_layer])
        # Interaction-1 sentences come from the held-out half: never seen when fitting.
        self.sources = {}
        for split, (pos_slice, neg_slice) in {"dev": (slice(0, 8), slice(0, 8)),
                                              "test": (slice(8, None), slice(8, None))}.items():
            pos_idx, neg_idx = pos_held[pos_slice], neg_held[neg_slice]
            self.sources[split] = {
                "positive": [data.EXPLICIT_POSITIVE[i] for i in pos_idx],
                "negative": [data.EXPLICIT_NEGATIVE[i] for i in neg_idx],
                "positive_z": self.reader.read_state(self.pos[read_layer][pos_idx]),
                "negative_z": self.reader.read_state(self.neg[read_layer][neg_idx]),
            }
        self._writers = {}

    def writer(self, layer: int):
        """(unit write direction at `layer`, class gap at `layer`)."""
        if layer not in self._writers:
            positive, negative = self.pos[layer][self.pos_fit], self.neg[layer][self.neg_fit]
            gap = float((positive.mean(0) - negative.mean(0)).norm())
            self._writers[layer] = (LatentWriter.from_direction(fit_direction(positive, negative)), gap)
        return self._writers[layer]

    def logits(self, prompt, layer, writer, z, alpha):
        return self.transformer.next_token_logits(
            prompt, layer, lambda hidden: writer.write_state(hidden, float(z), alpha))


def signed_shifts(exp, targets, layer, strength, baselines):
    """Mean signed shift and KL per source for each target. Signed: + means toward the source's sentiment."""
    split = exp.sources["dev"]
    writer, gap = exp.writer(layer)
    alpha = strength * gap
    shifts, kls = [], []
    for target in targets:
        reference_logits, reference_score = baselines[target]
        for label, sign in (("positive", 1.0), ("negative", -1.0)):
            for z in split[label + "_z"]:
                logits = exp.logits(target, layer, writer, z, alpha)
                shifts.append(sign * (exp.scorer(logits) - reference_score))
                kls.append(compare_logits(reference_logits, logits).kl_divergence)
    return float(np.mean(shifts)), float(np.mean(kls))


def baseline_logits(exp, targets):
    result = {}
    for target in targets:
        logits = exp.transformer.next_token_logits(target)
        result[target] = (logits, exp.scorer(logits))
    return result


def choose_config(exp, write_layers, log):
    baselines = baseline_logits(exp, data.DEV_TARGETS)
    grid = []
    for layer in write_layers:
        for strength in STRENGTHS:
            shift, kl = signed_shifts(exp, data.DEV_TARGETS, layer, strength, baselines)
            ratio = fluency(exp, layer, strength, DEV_NEUTRAL)["worst_ratio"] if kl <= KL_BUDGET else None
            grid.append({"write_layer": layer, "strength": strength, "dev_signed_shift": shift,
                         "dev_kl": kl, "dev_perplexity_ratio": ratio})
            log(f"  dev write_layer={layer:>2} strength={strength:<4} shift={shift:+.3f} kl={kl:.3f} ppl_ratio={ratio}")
    allowed = [row for row in grid if row["dev_kl"] <= KL_BUDGET and row["dev_signed_shift"] > 0
               and row["dev_perplexity_ratio"] <= DEV_PERPLEXITY_BUDGET]
    best = max(allowed, key=lambda row: row["dev_signed_shift"]) if allowed else min(grid, key=lambda row: row["dev_kl"])
    return best, grid


def evaluate_main(exp, layer, strength):
    """Relay vs controls on TEST targets with TEST sources."""
    split = exp.sources["test"]
    writer, gap = exp.writer(layer)
    alpha = strength * gap
    size = exp.transformer.hidden_size
    random_writers = [LatentWriter.from_direction(random_direction(size, seed)) for seed in (11, 12, 13)]
    per_target = {name: [] for name in ("relay", "random_direction", "wrong_label")}
    kls, raw_positive, raw_negative, hits = [], [], [], []
    for target in data.TEST_TARGETS:
        base_logits = exp.transformer.next_token_logits(target)
        base_score = exp.scorer(base_logits)
        rows = {name: [] for name in per_target}
        for label, sign, other in (("positive", 1.0, "negative"), ("negative", -1.0, "positive")):
            for index, z in enumerate(split[label + "_z"]):
                logits = exp.logits(target, layer, writer, z, alpha)
                shift = exp.scorer(logits) - base_score
                rows["relay"].append(sign * shift)
                (raw_positive if sign > 0 else raw_negative).append(shift)
                hits.append(shift * sign > 0)
                kls.append(compare_logits(base_logits, logits).kl_divergence)
                # Control: same z, same strength, but a random write direction.
                shifts = [exp.scorer(exp.logits(target, layer, rw, z, alpha)) - base_score
                          for rw in random_writers]
                rows["random_direction"].append(sign * float(np.mean(shifts)))
                # Control: the state from a source with the OPPOSITE sentiment.
                wrong_z = split[other + "_z"][index % len(split[other + "_z"])]
                wrong = exp.scorer(exp.logits(target, layer, writer, wrong_z, alpha)) - base_score
                rows["wrong_label"].append(sign * wrong)
        for name in per_target:
            per_target[name].append(float(np.mean(rows[name])))
    relay = np.array(per_target["relay"])
    random_ = np.array(per_target["random_direction"])
    result = {"n_targets": len(relay), "n_sources": len(split["positive"]) + len(split["negative"])}
    for name, values in per_target.items():
        mean, low, high = bootstrap_ci(values)
        result[name] = {"mean_signed_shift": mean, "ci95": [low, high], "p_value": sign_flip_p(values)}
    mean, low, high = bootstrap_ci(relay - random_)
    result["relay_minus_random"] = {"mean": mean, "ci95": [low, high], "p_value": sign_flip_p(relay - random_)}
    result["mean_shift_positive_sources"] = float(np.mean(raw_positive))
    result["mean_shift_negative_sources"] = float(np.mean(raw_negative))
    result["fraction_shifted_toward_source"] = float(np.mean(hits))
    result["mean_kl"] = float(np.mean(kls))
    return result


def dose_response(exp, layer, strength):
    writer, gap = exp.writer(layer)
    base = {t: exp.scorer(exp.transformer.next_token_logits(t)) for t in data.TEST_TARGETS}

    def mean_shift(z, alpha):
        return float(np.mean([exp.scorer(exp.logits(t, layer, writer, z, alpha)) - base[t]
                              for t in data.TEST_TARGETS]))

    by_z = [mean_shift(z, strength * gap) for z in Z_GRID]
    by_strength = [mean_shift(1.0, k * gap) for k in STRENGTH_GRID]
    within_budget = [k for k in STRENGTH_GRID if k <= max(strength, 1.0)]
    return {
        "z_values": list(Z_GRID), "shift_by_z": by_z, "spearman_z": spearman(Z_GRID, by_z),
        "strength_values": list(STRENGTH_GRID), "shift_by_strength": by_strength,
        "spearman_strength_up_to_chosen": spearman(within_budget, by_strength[:len(within_budget)]),
    }


def persistence(exp, layer, strength, beta):
    split = exp.sources["test"]
    writer, gap = exp.writer(layer)
    relay = LatentRelay(exp.transformer, exp.read_layer, layer, reader=exp.reader, writer=writer,
                        beta=beta, alpha=strength * gap)
    targets = data.TEST_TARGETS
    base = {t: exp.scorer(exp.transformer.next_token_logits(t)) for t in targets}
    sums = {n: [] for n in PERSISTENCE_STEPS}
    states = {n: [] for n in PERSISTENCE_STEPS}
    for label, sign in (("positive", 1.0), ("negative", -1.0)):
        for source in split[label][:6]:
            relay.reset()
            relay.remember(source)
            for step in range(max(PERSISTENCE_STEPS) + 1):
                if step:
                    relay.remember(data.NEUTRAL_TURNS[step - 1])  # an unrelated turn
                if step in sums:
                    states[step].append(sign * float(relay.z))
                    sums[step].append(np.mean([sign * (exp.scorer(relay.next_token_logits(t)) - base[t])
                                               for t in targets]))
    first = float(np.mean(sums[0]))
    return {
        "beta": beta,
        "steps": list(PERSISTENCE_STEPS),
        "signed_z": [float(np.mean(states[n])) for n in PERSISTENCE_STEPS],
        "signed_shift": [float(np.mean(sums[n])) for n in PERSISTENCE_STEPS],
        "fraction_of_first": [float(np.mean(sums[n]) / first) if first else None for n in PERSISTENCE_STEPS],
        "ideal_decay": [beta ** n for n in PERSISTENCE_STEPS],
    }


def fluency(exp, layer, strength, sentences=REPORT_NEUTRAL):
    writer, gap = exp.writer(layer)
    alpha = strength * gap

    def perplexity(z):
        total, count = 0.0, 0
        for sentence in sentences:
            if z == 0:
                nll, n = exp.transformer.sequence_nll(sentence)
            else:
                nll, n = exp.transformer.sequence_nll(
                    sentence, layer, lambda h: writer.write_state(h, z, alpha))
            total, count = total + nll, count + n
        return math.exp(total / count)

    normal = perplexity(0)
    return {"normal": normal, "z_plus_1": perplexity(1.0), "z_minus_1": perplexity(-1.0),
            "worst_ratio": max(perplexity(1.0), perplexity(-1.0)) / normal}


def generation(exp, layer, strength):
    split = exp.sources["test"]
    writer, gap = exp.writer(layer)
    relay = LatentRelay(exp.transformer, exp.read_layer, layer, reader=exp.reader, writer=writer,
                        beta=0.0, alpha=strength * gap)
    prompts = data.TEST_TARGETS[::3]

    def continuation(prompt, source):
        relay.reset()
        if source:
            relay.remember(source)
        return relay.generate(prompt, GENERATION_TOKENS)[len(prompt):]

    correct = ties = 0
    samples = []
    for prompt in prompts:
        pos = [continuation(prompt, s) for s in split["positive"][:GENERATION_SOURCES]]
        neg = [continuation(prompt, s) for s in split["negative"][:GENERATION_SOURCES]]
        pos_score = np.mean([lexicon_score(t) for t in pos])
        neg_score = np.mean([lexicon_score(t) for t in neg])
        correct += pos_score > neg_score
        ties += pos_score == neg_score
        samples.append({"prompt": prompt, "no_state": continuation(prompt, None),
                        "positive_state": pos[0], "negative_state": neg[0],
                        "positive_lexicon": float(pos_score), "negative_lexicon": float(neg_score)})
    return {"prompts": len(prompts), "correct_direction": correct / len(prompts),
            "ties": ties / len(prompts), "samples": samples}


def write_layers_for(layer_count: int) -> list[int]:
    return sorted({max(1, round(layer_count * f)) for f in (0.2, 0.35, 0.5, 0.65, 0.8)})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--read-layer", type=int, help="default: best layer from the Phase 7 sweep")
    parser.add_argument("--output", type=Path, help="default: results/phase8_<model>.json")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(4)
    log = lambda message: print(message, flush=True)
    transformer = FrozenTransformer(args.model, args.device)
    output = args.output or Path(f"results/phase8_{short_name(args.model)}.json")

    read_layer = args.read_layer
    if read_layer is None:
        pos = collect_activations(transformer, data.EXPLICIT_POSITIVE)
        neg = collect_activations(transformer, data.EXPLICIT_NEGATIVE)
        ip = collect_activations(transformer, data.IMPLICIT_POSITIVE)
        ineg = collect_activations(transformer, data.IMPLICIT_NEGATIVE)
        neutral = collect_activations(transformer, data.NEUTRAL_CENTER)
        read_layer = select_read_layer(
            sweep_layers(pos, neg, ip, ineg, neutral, range(1, transformer.layer_count + 1)))
    log(f"read layer: {read_layer}")
    exp = Experiment(transformer, read_layer)

    log("choosing write layer and strength on DEV prompts...")
    chosen, grid = choose_config(exp, write_layers_for(transformer.layer_count), log)
    layer, strength = chosen["write_layer"], chosen["strength"]
    log(f"chosen: write_layer={layer}, strength={strength} class gaps")

    log("evaluating on TEST prompts...")
    main_result = evaluate_main(exp, layer, strength)
    log("dose-response...")
    dose = dose_response(exp, layer, strength)
    log("persistence...")
    persist = [persistence(exp, layer, strength, beta) for beta in PERSISTENCE_BETAS]
    log("fluency...")
    fluent = fluency(exp, layer, strength)
    log("generation...")
    generated = generation(exp, layer, strength)

    relay, random_, wrong = main_result["relay"], main_result["random_direction"], main_result["wrong_label"]
    slow = next(item for item in persist if item["beta"] == 0.8)
    criteria = {name: bool(value) for name, value in {
        "relay_shifts_toward_source_significantly": relay["ci95"][0] > 0 and relay["p_value"] < 0.05,
        "beats_random_direction": main_result["relay_minus_random"]["ci95"][0] > 0,
        "wrong_label_state_shifts_opposite": wrong["mean_signed_shift"] < 0,
        "dose_response_monotonic_in_z": dose["spearman_z"] >= 0.9,
        "dose_response_monotonic_in_strength": dose["spearman_strength_up_to_chosen"] >= 0.9,
        "persists_one_and_three_turns": slow["signed_shift"][1] > 0 and slow["signed_shift"][2] > 0,
        "kl_within_budget": main_result["mean_kl"] <= KL_BUDGET,
        "perplexity_within_budget": fluent["worst_ratio"] <= PERPLEXITY_BUDGET,
        "generation_direction_at_least_70": generated["correct_direction"] >= 0.7,
    }.items()}
    report = {
        "model": args.model, "read_layer": read_layer, "chosen": chosen,
        "budgets": {"kl": KL_BUDGET, "perplexity_ratio": PERPLEXITY_BUDGET,
                    "dev_perplexity_ratio": DEV_PERPLEXITY_BUDGET},
        "dev_grid": grid, "main": main_result, "dose_response": dose,
        "persistence": persist, "fluency": fluent, "generation": generated, "criteria": criteria,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    log(f"\nrelay signed shift: {relay['mean_signed_shift']:+.3f} (95% CI {relay['ci95'][0]:+.3f}..{relay['ci95'][1]:+.3f}, p={relay['p_value']:.4f})")
    log(f"random direction:   {random_['mean_signed_shift']:+.3f}   wrong-label state: {wrong['mean_signed_shift']:+.3f}")
    for name, passed in criteria.items():
        log(f"  {'PASS' if passed else 'FAIL'}  {name}")
    log(f"saved {output}")


if __name__ == "__main__":
    main()

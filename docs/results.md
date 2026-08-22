# Results (Phases 7 and 8, plus the memory task)

Short version: **a single stored number, read from one sentence and written back into a
later, unrelated prompt, reliably steers the model toward that sentence's sentiment.**
The effect scales with the number and with strength, fades at the rate you set, and
random or opposite-sentiment controls do nothing useful. It is not free: the fluency cost
is real and, on the smaller model, slightly over budget. Details and caveats below.

Models: `pythia-160m` (12 layers) and `pythia-410m` (24 layers), CPU, fp32.
Plain-language explanations of every term are in [how-it-works.md](how-it-works.md).

## Phase 7: does the direction read sentiment?

A direction is fitted on half of 96 explicit sentences and scored on the other half, averaged
over 5 random splits. "Implicit" sentences describe good or bad events without emotion words
("She was laid off after twenty years") and were never used to choose a layer.

| Model | Chosen read layer | Held-out accuracy | AUROC | Implicit accuracy | Shuffled-label control | Random-direction control |
|---|---|---|---|---|---|---|
| pythia-160m | 12 | 91.2% | 0.964 | 65.0% | 53.7% | 49.7% |
| pythia-410m | 14 | 97.1% | 0.997 | 67.9% | 55.8% | 49.3% |

- Passes: held-out accuracy at least 85%, and both controls near 50%.
- **Fails: implicit accuracy (target 75%).** The direction mostly tracks emotional wording,
  not the underlying situation. It is a sentiment-word reader more than a sentiment-meaning reader.
- Accuracy rises with depth: layers 1 to 3 are near chance, and the signal grows through the middle layers.

## Phase 8: does a stored number steer a later interaction?

Setup: read `z` from a held-out sentence, discard the text, then steer 36 neutral test prompts
(for example "Overall, the movie turned out to be") with `z`. The write layer and strength were chosen on 12 separate dev
prompts under a KL budget (0.5 nats) and a perplexity budget (1.25x). "Shift" is the change in
log P(positive words) minus log P(negative words); signed so that positive means "toward the source".
CIs are 95% bootstrap intervals over test prompts.

| | pythia-160m | pythia-410m |
|---|---|---|
| Read layer / write layer | 12 / 8 | 14 / 16 |
| Strength (class gaps) | 8.0 | 1.0 |
| **Signed shift, relay** | **+1.65** (1.59 to 1.72) | **+2.44** (2.29 to 2.60) |
| Signed shift, random write direction | +0.08 | -0.07 |
| Relay minus random (CI) | 1.51 to 1.63 | 2.35 to 2.67 |
| (source, prompt) pairs shifted the right way | 96.9% | 100% |
| Mean shift, positive sources / negative sources | +2.00 / -1.31 | +2.49 / -2.39 |
| Mean KL from unsteered distribution | 0.073 | 0.119 |
| Perplexity ratio, held-out neutral text (budget 1.25) | **1.35 (over budget)** | 1.14 |
| Generated text shifted the right way (12 prompts) | 92% | 100% |

**Dose-response.** Shift at z = -1, -0.5, 0, +0.5, +1 is -2.44, -1.75, 0, +2.10, +3.62 on 160m and
-3.21, -1.98, 0, +2.24, +4.18 on 410m. Both are perfectly monotonic (Spearman 1.0), and
shift also rises monotonically with strength.

**Persistence.** After the source sentence, 1, 3, and 5 unrelated neutral turns are stored.
The effect fades at roughly the rate set by `beta` (the fraction left, versus the ideal `beta^n`):

| beta | turns later | 160m | 410m | ideal |
|---|---|---|---|---|
| 0.5 | 1 / 3 / 5 | 0.38 / 0.13 / 0.03 | 0.50 / 0.13 / 0.03 | 0.50 / 0.12 / 0.03 |
| 0.8 | 1 / 3 / 5 | 0.75 / 0.52 / 0.32 | 0.80 / 0.52 / 0.33 | 0.80 / 0.51 / 0.33 |

So the memory persists and its half-life is tunable.

**Pass/fail against the targets set before running:** 410m passes 9 of 9. 160m passes 8 of 9
and fails the perplexity budget (see caveats).

## Memory task (learned relay)

A tiny trained read/write pair (6,149 trainable parameters, frozen model) stores one of four
codes and recalls it in a separate prompt. Evaluated on held-out wording for both the
statement and the question (16 examples, so each example is 6.25 points):

| Condition | Accuracy |
|---|---|
| **Relay** | **100%** |
| Zero state (no memory) | 25% (chance) |
| State from the wrong code | 0% |
| Text in context (not the relay) | 87.5% |

Before the fix described below the relay scored 25% (chance). The read map's `tanh` had
saturated, so the state was the same for every input. Centering and scaling the read input
using training statistics, and a lower learning rate, fixed it. A probe showed the code was
linearly readable at every layer, so the information was always there; it was being lost on read.

## Caveats and limitations

1. **Fluency cost is real.** Steering raises perplexity. On 160m the final check on held-out
   neutral sentences measured 1.35x against 1.21x on the dev sentences used to choose the setting,
   so the budget was missed. On 160m the best setting also sits at the top of the strength grid.
2. **The wrong-label control is a mirror, not an independent test.** Each positive source is
   paired with a negative one, so the wrong-label result is exactly the negative of the
   relay result by construction. The random-direction control and dose-response curve are the
   independent evidence.
3. **Asymmetric on 160m.** Positive sources shift +2.00 but negative sources only -1.31.
4. **Sentiment is measured by word lists.** The score uses single-token positive and negative
   words; generation is judged with a crude word-stem lexicon over 12 prompts. Generated samples are
   saved in `results/phase8_*.json` for reading by eye.
5. **Small data.** 48 explicit sentences per class, one fixed fit/held-out split in Phase 8,
   9 test subjects. Intervals reflect variation across test prompts, not across fitted directions.
6. **Implicit sentiment fails** (see Phase 7).
7. **Scalar steering is a known technique.** The new part here is persisting the number
   across separate interactions and measuring the decay, not the activation steering itself.
8. **Memory task** has 16 test examples and a simple balanced four-way task.

## Reproduce

```bash
python -m experiments.phase7_sentiment [--model EleutherAI/pythia-410m]
python -m experiments.phase8_persistence [--model EleutherAI/pythia-410m]
python -m experiments.memory_task
```

Outputs go to `results/` (git-ignored). Phase 8 takes about 10 minutes on 160m and about 25 on 410m.

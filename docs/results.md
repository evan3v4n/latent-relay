# Results (Phases 7 and 8, plus the memory task)

Short version: **a single stored number, read from one sentence and written back into a
later, unrelated prompt, reliably steers the model toward that sentence's sentiment.**
On `pythia-410m` every check passes. On `pythia-160m` the effect is real but one-sided and
weaker, and two of nine Phase 8 checks fail. Details and caveats below.

Models: `pythia-160m` (12 layers) and `pythia-410m` (24 layers), CPU, fp32.
Plain-language explanations of every term are in [how-it-works.md](how-it-works.md).

## Phase 7: does the direction read sentiment?

A direction is fitted on half of 96 explicit sentences and scored on the other half, averaged
over 5 random splits. The zero point is the mean of 20 separate neutral sentences, so "neutral
text reads as 0". "Implicit" sentences describe good or bad events without emotion words
("She was laid off after twenty years") and were never used to choose a layer.

| Model | Chosen read layer | Held-out accuracy | AUROC | Implicit accuracy | Shuffled-label control | Random-direction control |
|---|---|---|---|---|---|---|
| pythia-160m | 12 | 86.2% | 0.964 | **70.4%** | 47.9% | 49.3% |
| pythia-410m | 14 | 97.9% | 0.997 | 77.5% | 57.1% | 48.8% |

- 410m passes all four checks. 160m passes three and **misses implicit accuracy** (target 75%).
- **Why the zero point matters.** Using the midpoint between the positive and negative examples
  as zero, implicit accuracy was 65% (160m) and 68% (410m) even though rank-based AUROC on the
  implicit sentences was already 0.86 and 0.95: the direction was right but the threshold was
  shifted, because implicit sentences all landed on one side of the midpoint. Centering on neutral text
  fixed most of that on 410m. It cost 160m some explicit accuracy (91% to 86%).
- Mean-pooling activations over all words instead of using the last word was worse on both models.
- Accuracy rises with depth: layers 1 to 3 are near chance and the signal builds through the middle layers.

## Phase 8: does a stored number steer a later interaction?

Setup: read `z` from a held-out sentence, discard the text, then steer 36 neutral test prompts
(for example "Overall, the movie turned out to be") with `z`. The write layer and strength were
chosen on 12 separate dev prompts, under a KL budget (0.5 nats) and a stricter dev perplexity
budget (1.15x), using their own set of neutral sentences. "Shift" is the change in
log P(positive words) minus log P(negative words), signed so that positive means "toward the source".
CIs are 95% bootstrap intervals over test prompts.

| | pythia-160m | pythia-410m |
|---|---|---|
| Read layer / write layer | 12 / 6 | 14 / 16 |
| Strength (class gaps) | 8.0 | 1.0 |
| **Signed shift, relay** | **+1.03** (0.95 to 1.10) | **+2.48** (2.34 to 2.62) |
| Signed shift, random write direction | -0.04 | -0.07 |
| Relay minus random (CI) | 1.01 to 1.13 | 2.41 to 2.70 |
| (source, prompt) pairs shifted the right way | 83.9% | 100% |
| Mean shift, positive sources / negative sources | +1.79 / -0.26 | +2.88 / -2.09 |
| Mean KL from unsteered distribution | 0.059 | 0.117 |
| Perplexity ratio, held-out neutral text (budget 1.25) | 1.16 | 1.15 |
| Generated text shifted the right way (12 prompts) | **33%** | 92% |
| Checks passed | 7 of 9 | 9 of 9 |

**160m failures.** (1) Dose-response is not monotonic in z: the shift is -0.87 at z = -1 and -0.91
at z = -0.5. (2) Generation direction is 33%, below the 70% target. Both come from the same
cause: at the chosen layer, negative states barely move the model (-0.26 on average) while positive
states do (+1.79). An earlier 160m setting at a different layer gave a more balanced effect (+2.0 / -1.3)
but exceeded the perplexity budget, so the stricter selection traded balance for fluency.

**Dose-response, 410m.** Shift at z = -1, -0.5, 0, +0.5, +1 is -3.21, -1.98, 0, +2.24, +4.18
(Spearman 1.0), and rises monotonically with strength.

**Persistence.** After the source sentence, 1, 3, and 5 unrelated neutral turns are stored.
The fraction of the original effect that remains, versus the ideal `beta^n`:

| beta | turns later | 160m | 410m | ideal |
|---|---|---|---|---|
| 0.5 | 1 / 3 / 5 | 0.48 / 0.13 / 0.03 | 0.51 / 0.13 / 0.03 | 0.50 / 0.12 / 0.03 |
| 0.8 | 1 / 3 / 5 | 0.77 / 0.52 / 0.34 | 0.81 / 0.52 / 0.33 | 0.80 / 0.51 / 0.33 |

The memory persists and its half-life is tunable.

## Memory task (learned relay)

A tiny trained read/write pair (6,149 trainable parameters, frozen model) stores one of four
codes and recalls it in a separate prompt. Evaluated on held-out wording for both the
statement and the question (16 examples, so each example is 6.25 points; 160m only):

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

1. **160m is the weak model.** Implicit sentiment (70%), balance between positive and negative
   steering, dose-response, and generation all fall short there. 410m passes everything, so
   treat the 160m results as a lower bound on what this approach does at small scale.
2. **Fluency cost is real everywhere.** Steering raises perplexity by about 15% at the chosen settings.
   Settings are chosen with a 1.15x margin on a dev set, because a 1.25x budget checked on the same small dev
   set was missed on held-out text in an earlier run.
3. **The wrong-label control is a mirror, not an independent test.** Each positive source is
   paired with a negative one, so the wrong-label result is exactly the negative of the
   relay result by construction. The random-direction control and dose-response curve are the
   independent evidence.
4. **Sentiment is measured by word lists.** The score uses single-token positive and negative
   words; generation is judged with a crude word-stem lexicon over 12 prompts. Generated samples are
   saved in `results/phase8_*.json` for reading by eye.
5. **Small data.** 48 explicit sentences per class, one fixed fit/held-out split in Phase 8,
   9 test subjects. Intervals reflect variation across test prompts, not across fitted directions.
6. **The design changed after seeing results.** The neutral zero point and the stricter dev
   perplexity margin were added after earlier runs missed targets. Test prompts were not used
   to choose either, but the targets were not fully fixed in advance.
7. **Scalar steering is a known technique.** The new part here is persisting the number
   across separate interactions and measuring the decay, not the activation steering itself.
8. **Memory task** has 16 test examples, a simple balanced four-way task, and was only run on 160m.

## Reproduce

```bash
python -m experiments.phase7_sentiment [--model EleutherAI/pythia-410m]
python -m experiments.phase8_persistence [--model EleutherAI/pythia-410m]
python -m experiments.memory_task
```

Outputs go to `results/` (git-ignored). Phase 8 takes about 10 minutes on 160m and about 25 on 410m.

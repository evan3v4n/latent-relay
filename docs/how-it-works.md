# How Latent Relay works (plain-language guide)

## The idea in one paragraph

A language model normally remembers a conversation only by re-reading the text. Latent
Relay tests a different channel: after one interaction, we squeeze what the model "thought"
into a tiny number (or a few numbers), throw the text away, and feed that number back into
the model during a later, unrelated interaction. If the later answer changes in a
predictable way, the number carried information across, with no text involved.

```text
Interaction 1                                   Interaction 2 (fresh context)
"Something amazing happened today."             "The movie turned out to be ..."
        │                                               │
  frozen model ──> READ ──> z = +0.8  ──────────> WRITE into the model ──> next word
                           (the only thing kept)         (nudges its internal state)
```

The model's own weights are never changed. Everything here is a small add-on around it.

## Vocabulary

| Term | Meaning |
|---|---|
| **Frozen model** | A Pythia language model whose weights never change. |
| **Hidden state / activation** | The list of numbers the model holds internally at each layer for each word. |
| **Direction** | A line through that number space, such as "the positive-vs-negative sentiment line". |
| **Reader** | Measures how far an activation sits along a direction and returns `z`, a number between -1 and +1. `0` means neutral. |
| **State `z`** | The stored number. It is the only thing carried between interactions. |
| **Writer** | Adds `z × strength` along a direction to the model's activations in a later interaction. |
| **Read layer / write layer** | Which layer the reader looks at, and which layer the writer pushes on. |
| **Class gap** | The distance between "typical positive" and "typical negative" activations at a layer. Strength is measured in class gaps so that "1.0" means the same thing at every layer. |
| **β (beta)** | How much old state is kept when a new interaction arrives: `z_new = β·z_old + (1-β)·z_observed`. |
| **Held-out** | Data that was not used to fit or tune anything. All headline numbers use held-out data. |

## What each phase does

- **Phase 7, give `z` a meaning.** Fit a direction from example positive and negative
  sentences, then check whether it reads the sentiment of sentences it has never seen.
  (`experiments/phase7_sentiment.py`)
- **Phase 8, test persistent memory.** Store `z` from one sentence, delete the text, then
  steer a neutral prompt with it and measure the change in the next-word probabilities.
  (`experiments/phase8_persistence.py`)
- **Memory task (a learned relay).** A tiny trained read/write pair stores which of four
  secret codes (A–D) it saw and recalls it later. (`experiments/memory_task.py`)

## How "did it work?" is measured

For Phase 8 the model is asked to continue prompts like "Overall, the movie turned out to
be". We add up the probability of positive words (good, great, wonderful, ...) and of
negative words (bad, terrible, awful, ...) and take the log ratio. The **shift** is how
much that score moves compared with no stored state. A **signed shift** flips the sign for
negative sources, so a positive number always means "moved toward the stored sentiment".

Every result is compared against controls that should do nothing:

| Control | What it rules out |
|---|---|
| **Random direction** | The effect is just "any push on the activations changes the answer". |
| **Wrong-label state** (state from the opposite sentiment) | Should shift the *opposite* way, proving the content of `z` matters. |
| **Reset / zero state** | Without a stored state nothing should change. |
| **Shuffled-label direction** (Phase 7) | The direction is not accidentally good at separating the sentences. |
| **Random direction** (Phase 7) | Chance level is about 50%. |

## Guard rails against fooling ourselves

1. **Held-out sentences.** The direction is fitted on one half of the sentences; sources for
   Phase 8 come from the other half.
2. **Held-out prompts.** The write layer and strength are chosen on 3 "dev" subjects. The
   final numbers use 9 different "test" subjects.
3. **Implicit-sentiment check.** Sentences like "She was laid off after twenty years" contain
   no obvious emotion words. They are never used to choose a layer.
4. **Side effects are budgeted.** Steering must not wreck the model: the average change in
   the next-word distribution (KL) and the perplexity on neutral text must stay under fixed
   limits.
5. **Confidence intervals and permutation tests** on every headline effect.

## Running things

```bash
python -m experiments.phase7_sentiment                  # ~10 s on pythia-160m
python -m experiments.phase8_persistence                # ~10 min on pythia-160m
python -m experiments.memory_task                       # ~40 s
python -m experiments.phase7_sentiment --model EleutherAI/pythia-410m
python -m unittest discover -s tests -t .               # fast unit tests, no model download
```

Results are written to `results/` (not committed). The headline numbers are recorded in
[results.md](results.md).

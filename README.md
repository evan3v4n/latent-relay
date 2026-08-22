# Latent Relay

**Can a language model carry a memory between conversations in a single number, with no text?**

Latent Relay reads one small number out of a frozen language model's internal activations
after an interaction, throws the text away, and writes that number back into the model during a
later, unrelated interaction. If the later output changes in a predictable way, the number carried
information.

```text
Interaction 1                                 Interaction 2 (fresh context)
"Something amazing happened today."           "Overall, the movie turned out to be ..."
        │                                              │
  frozen model ──> READ ──> z = +0.8 ───────> WRITE into the model ──> more positive words
                          (the only thing kept)
```

Model weights are never changed. See [docs/how-it-works.md](docs/how-it-works.md) for a
plain-language walkthrough and glossary.

## Headline results

On held-out sentences and held-out prompts, with controls ([full details and caveats](docs/results.md)):

| | pythia-160m | pythia-410m |
|---|---|---|
| Sentiment direction, held-out accuracy | 91% | 97% |
| Effect of a stored positive/negative number on a later prompt | +1.65 (CI 1.59 to 1.72) | +2.44 (CI 2.29 to 2.60) |
| Same effect with a random write direction (control) | +0.08 | -0.07 |
| Effect fades after unrelated turns at the set `beta` rate | yes | yes |
| Phase 8 checks passed | 8 of 9 | 9 of 9 |
| Learned four-code memory task, held-out recall (chance 25%) | 100% | not run |

Known weak points: the 160m run goes over the fluency (perplexity) budget, and the direction
reads emotional wording better than situations without emotion words (65% to 68% on those).

## Roadmap status

| Phase | What | Status |
|---|---|---|
| 0–2 | Trace Pythia activations, read a bounded scalar | done |
| 3 | Smooth the state across independent interactions (`LatentState`) | done |
| 4 | Write the state into a later layer with a temporary hook | done |
| 5 | Compare identical prompts at z = -1, 0, +1 | done |
| 6 | Sweep read/write layers | done |
| 7 | Give the number a meaning (fitted sentiment direction) | done; implicit-sentiment target missed |
| 8 | Persistent memory across interactions | done; 410m passes all checks, 160m misses fluency budget |
| 10 (early) | Trained low-rank relay | working on a four-code memory task |
| 9 | Multi-dimensional state | not started |

## Run it

Python 3.10+. The first run downloads the model (about 600 MB for 160m).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m experiments.phase7_sentiment       # ~10 s: does the direction read sentiment?
python -m experiments.phase8_persistence     # ~10 min: does a stored number steer later prompts?
python -m experiments.memory_task            # ~40 s: learned four-code memory
python -m unittest discover -s tests -t .    # fast tests, no model download
```

Add `--model EleutherAI/pythia-410m` to the Phase 7 and 8 commands to repeat them on the larger
model. Results are written to `results/` as JSON.

Earlier experiments still work:

```bash
python main.py                                       # Phase 6 layer sweep -> results/phase6.csv
python -m experiments.inspect_activations --layers 0 4 8 12
python -m experiments.write_latent --read-layer 12 --write-layer 8
python -m experiments.prove_effect --write-layer 8 --alpha 1
python -m experiments.learn_direction
```

## Using the relay in code

```python
from model import FrozenTransformer
from relay import CalibratedReader, LatentWriter
from session import LatentRelay

relay = LatentRelay(transformer, read_layer, write_layer,
                    reader=reader, writer=writer, beta=0.8, alpha=alpha)
relay.remember("Something amazing happened today.")   # reads and stores z
relay.next_token_logits("The movie turned out to be")  # steered by z, no earlier text
relay.generate("The movie turned out to be", max_new_tokens=20)
relay.reset()                                          # start a fresh session
```

One `LatentRelay` per conversation. `experiments/phase8_persistence.py` shows how to fit a reader
and writer from example sentences.

## Repository map

| File | Purpose |
|---|---|
| `model.py` | Frozen Pythia wrapper: trace activations, temporary write hook, scoring |
| `relay.py` | Reader, writer, smoothed state, direction fitting |
| `session.py` | `LatentRelay`: the stateful product-shaped interface |
| `trainable.py` | Small trained read/write maps for the memory task |
| `sentiment_data.py` | Hand-written sentences, prompts, and word lists |
| `stats.py` | Bootstrap intervals, permutation test, AUROC |
| `experiments/` | One script per phase |
| `tests/` | Unit tests |
| `outline.md` | The original project plan |
| `docs/` | Guide and results |

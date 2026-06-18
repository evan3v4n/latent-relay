# Latent Relay

A small experimental harness that reads, persists, and writes a scalar state
through frozen Pythia activations. Phases 0–7 now cover the relay mechanism,
causal intervention, layer sweeps, and a first learned semantic direction.

The reader normalizes both its direction and each activation before projection.
This measures directional alignment and prevents Pythia's large activation
norms from forcing every state to `-1` or `+1`.

## Run locally

Python 3.10+ is recommended. The first run downloads the model (roughly 600 MB).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The default experiment sweeps the layer pairs from the project outline and
writes its effect measurements to `results/phase6.csv`.

Change the layer pairs, strength, or prompts:

```bash
python main.py --pairs 3:3 3:8 6:8 --alpha 0.5
python main.py --source "A quiet morning" --target "Outside, the weather was"
```

Earlier experiments remain available:

```bash
python -m experiments.inspect_activations --layers 0 4 8 12
python -m experiments.read_latent --layer 8 --seed 42
python -m experiments.write_latent --read-layer 12 --write-layer 8
python -m experiments.prove_effect --write-layer 8 --alpha 1
```

`LatentReader`, `LatentState`, and `LatentWriter` are independent components.
`FrozenTransformer.next_token_logits()` owns the temporary layer hook and always
removes it after the forward pass. `compare_logits()` provides reusable effect
measurements for later read/write-layer sweeps.

## Roadmap status

- **Phases 0–2 — complete:** load Pythia, trace hidden states, and read a bounded scalar.
- **Phase 3 — complete:** smooth the scalar across independent interactions with `LatentState`.
- **Phase 4 — complete:** inject the state into a later transformer block with a temporary hook.
- **Phase 5 — complete:** compare identical prompts under `z=-1`, `0`, and `+1`.
- **Phase 6 — complete:** sweep read/write layers and save measurements to CSV.
- **Phase 7 — started:** fit a direction from positive/negative activation examples with `fit_direction()`.
- **Phase 8 — started:** use `LatentRelay` for a stateful session and token generation.

The next research milestone is not more plumbing: evaluate whether a learned
direction transfers from the source prompts to held-out target prompts. Then
compare the relay against text-only memory and a zero-state baseline, tune
read/write layers and `alpha`, and measure persistence, usefulness, and drift.

## Product shape

The final product is a small stateful inference layer around a frozen language
model. A host application keeps one `LatentRelay` per conversation. Each turn
can be remembered, which reads the turn's activation into a compact latent
state; later prompts are run through the same transformer with that state
written into an internal layer. The text context remains separate from the
latent channel, and `reset()` starts a fresh session.

```text
user turn ──> frozen transformer ──> reader ──> state z
                                              │
new prompt ─> frozen transformer ──> writer <─┘
                                └──> next-token distribution ──> response
```

At production quality, the API should add a trained direction artifact,
calibrated layer/strength settings, persistence policy, and evaluation logs.
The current implementation deliberately keeps those choices explicit so the
mechanism can be measured before it is packaged as a general memory system.

To try the learned-direction experiment:

```bash
python -m experiments.learn_direction
```

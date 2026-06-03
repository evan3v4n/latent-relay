# Latent Relay

A small experimental harness for building a persistent latent channel around a
frozen transformer. Phase 2 projects Pythia-160M's final-token activation onto
a reproducible random unit direction and applies `tanh` to produce a scalar
latent value in `[-1, 1]`.

## Run locally

Python 3.10+ is recommended. The first run downloads the model (roughly 600 MB).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The output reports a latent value for each example prompt. By default it reads
the final transformer block with random-direction seed `0`.

Read another layer, direction, or set of prompts:

```bash
python main.py --layer 8 --seed 42
python main.py "A quiet morning in Paris" "Markets rose after the report"
```

Run the Phase 1 comparison directly with:

```bash
python -m experiments.inspect_activations --layers 0 4 8 12
```

`FrozenTransformer.trace()` remains the activation boundary. `LatentReader` is
a small `torch.nn.Module`, so later phases can compose it with persistent state
and write interventions without coupling those pieces to experiment code.

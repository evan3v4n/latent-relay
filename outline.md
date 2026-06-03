## Goal

Build a small recurrent mechanism around a frozen transformer where the model maintains a **persistent latent state** across separate forward passes.

Instead of only passing information through text context, the model carries a scalar internal state:

\[
z_t
\]

That state is:

\[
\text{read from activations} \rightarrow \text{updated} \rightarrow \text{written back into later activations}
\]

The core loop eventually looks like:

\[
h_t \rightarrow z_t \rightarrow z_{t+1} \rightarrow h'_{t+1}
\]

with

\[
z_t = \tanh(u^\top h_t)
\]

and

\[
h'_{t+1} = h_{t+1} + \alpha z_t v
\]

where \(u\) is a **read direction**, \(v\) is a **write direction**, and \(h\) is a transformer hidden activation.

---

# Phase 0 — Understand the base model

You’re basically here now.

Use **Pythia-160M** as the experimental model because it’s small enough to run easily while still being a real transformer.

### Build

```text
text
 ↓
tokenizer
 ↓
Pythia
 ↓
logits
```

You should understand:

- tokens
- embeddings
- transformer layers
- hidden states
- logits
- generation
- tensor shapes

### Milestone

You can do:

```python
inputs = tokenizer(text, return_tensors="pt")

outputs = model(
    **inputs,
    output_hidden_states=True
)
```

and understand what comes back.

---

# Phase 1 — Read internal activations

Now inspect what the model is doing internally.

Pick a layer:

```python
h = outputs.hidden_states[layer]
```

Its shape will roughly be:

```text
[batch, sequence_length, 768]
```

Take the last token:

```python
h_last = h[:, -1, :]
```

giving:

```text
[1, 768]
```

This vector becomes the interface between the transformer and your relay.

### Experiment

Run:

```text
"The dog ran across the"
```

and:

```text
"The stock market crashed because"
```

Extract `h_last` for each.

Compare:

- norms
- cosine similarity
- different layers

You want intuition for what these activations actually look like.

### Milestone

You can reliably extract a **768-dimensional activation** from any prompt.

---

# Phase 2 — Build the latent reader

Now reduce the 768-dimensional hidden state into your scalar \(z\).

Create a direction:

\[
u \in \mathbb{R}^{768}
\]

Normalize it:

\[
\hat u = \frac{u}{||u||}
\]

Then:

\[
z = \tanh(\hat u^\top h)
\]

In code, conceptually:

```python
u = torch.randn(768)
u = u / u.norm()

z = torch.tanh(h_last @ u)
```

Now:

```text
768-dimensional activation
        ↓
      dot product
        ↓
       scalar
        ↓
       tanh
        ↓
    -1 <= z <= 1
```

### Experiment

Feed many prompts and record `z`.

For example:

```text
"I am extremely happy today."
"I just lost something important."
"The number seven is prime."
"Paris is in France."
```

At this point, a **random** \(u\) won't mean anything semantically.

That’s okay.

You are testing the mechanism.

### Milestone

You have:

```python
def read_state(h):
    return z
```

---

# Phase 3 — Make the state persistent

Right now, \(z\) disappears after every forward pass.

Change that.

Create something like:

```python
class LatentState:
    def __init__(self):
        self.z = 0.0
```

Then every interaction has access to the previous value:

```text
interaction 1
    ↓
read activation
    ↓
z₁

interaction 2
    ↓
previous z₁ still exists
```

You can start with the simplest update:

\[
z_{t+1} = \beta z_t + (1-\beta)\hat z_t
\]

where \(\hat z_t\) is the newly read state.

Example:

```python
z = beta * previous_z + (1 - beta) * new_z
```

If:

```text
β = 0
```

there is no memory.

If:

```text
β ≈ 1
```

the state changes very slowly.

### Milestone

Your model now has a value that persists across otherwise independent prompts.

---

# Phase 4 — Write the state back into the model

This is where **Latent Relay becomes interesting**.

Create another direction:

\[
v \in \mathbb{R}^{768}
\]

Normalize it.

Then modify an activation:

\[
h' = h + \alpha z v
\]

Conceptually:

```python
h_modified = h + alpha * z * v
```

Now the previous interaction can influence the next forward pass **without putting that information into the textual context**.

The architecture becomes:

```text
Prompt A
   ↓
Transformer
   ↓
hidden activation
   ↓
read direction u
   ↓
   z
   │
   │ persists
   ↓
Prompt B
   ↓
Transformer layer
   ↓
h + αzv
   ↓
remaining layers
   ↓
output
```

### Technical challenge

You’ll need to intervene inside the transformer.

Likely using:

- forward hooks, or
- editing/intercepting a transformer layer's output.

This will be the first somewhat tricky implementation step.

### Milestone

Changing \(z\) manually changes the model's output distribution.

---

# Phase 5 — Prove the relay actually affects behavior

Before trying to give \(z\) meaning, verify causality.

Run the **same exact prompt** with:

```text
z = -1
z = 0
z = +1
```

Everything else identical.

Measure how much the logits change.

For instance:

\[
D_{KL}(P(y|z=1) \; || \; P(y|z=0))
\]

You don't even need KL divergence initially.

You can inspect:

- top tokens
- token probabilities
- generated continuations

Example:

```text
Prompt:
"The person opened the door and saw"
```

Run it with different \(z\).

### Question

Does manipulating the latent channel actually influence model behavior?

If not, increase \(\alpha\) carefully or experiment with different layers.

---

# Phase 6 — Find good read/write layers

Now start doing actual experiments.

Try different combinations:

```text
read layer 4 → write layer 4
read layer 4 → write layer 8
read layer 8 → write layer 8
read layer 8 → write layer 10
```

Measure how strongly the intervention affects the output.

You may find that earlier layers have different effects from later layers.

Build a table:

| Read layer | Write layer |   α | Output effect |
| ---------- | ----------- | --: | ------------: |
| 3          | 3           | 0.1 |         small |
| 3          | 8           | 0.1 |        medium |
| 6          | 8           | 0.1 |        strong |

This starts turning the project into something empirical rather than purely conceptual.

---

# Phase 7 — Give the latent dimension meaning

So far \(u\) and \(v\) are random.

That's not very interesting.

Now learn or construct directions corresponding to something measurable.

For example:

```text
positive ↔ negative sentiment
certainty ↔ uncertainty
agreement ↔ disagreement
formal ↔ informal
```

Suppose you want sentiment.

Collect activations for:

```text
positive prompts
negative prompts
```

Compute:

\[
u =
\text{mean}(h_{\text{positive}})
-

\text{mean}(h_{\text{negative}})
\]

Normalize it.

Now:

\[
z = u^\top h
\]

might actually represent something interpretable.

This is basically a simple **linear probe / activation direction**.

---

# Phase 8 — Test persistent latent memory

Now you can run experiments like:

### Interaction 1

```text
User:
Something amazing happened today.
```

The relay reads:

```text
z = +0.7
```

Then remove the text entirely.

### Interaction 2

Fresh context:

```text
User:
Describe the weather.
```

But the model receives:

\[
h' = h + \alpha(0.7)v
\]

Compare against:

```text
z = 0
```

The key research question becomes:

> Can information from a prior interaction systematically influence future behavior through a tiny persistent latent channel?

That is the heart of the project.

---

# Phase 9 — Increase state dimensionality

Once the scalar mechanism works, generalize:

\[
z \in \mathbb{R}^k
\]

For example:

```text
k = 4
k = 8
k = 16
```

Then:

\[
z = U^\top h
\]

and

\[
h' = h + Vz
\]

where:

\[
U,V \in \mathbb{R}^{768 \times k}
\]

Now you've effectively added a tiny **recurrent latent memory** around a frozen transformer.

Architecture:

```text
        ┌──────── z_t ────────┐
        │                     │
        ↓                     │
input → transformer → h_t → read
          ↑
          │
         write
          ↑
        z_t
```

At that point the project becomes substantially more interesting.

---

# Phase 10 — Train the relay

Instead of manually choosing \(U\) and \(V\), train them while keeping Pythia frozen.

Trainable parameters:

```text
U
V
state update
```

Frozen:

```text
all Pythia weights
```

You could train on tasks where information has to survive across separated contexts.

Example:

```text
Turn 1:
Remember whether the secret bit is A or B.

<context removed>

Turn 2:
What was the secret bit?
```

Crucially, **the answer cannot exist in the text context**.

The only route is:

```text
hidden state
   ↓
latent relay
   ↓
persistent z
   ↓
later forward pass
```

That gives you a real test of whether latent recurrence is functioning as memory.

---

# Suggested repo structure

Keep the project simple:

```text
latent-relay/

├── README.md
├── requirements.txt
│
├── main.py
│
├── model.py
├── relay.py
│
├── experiments/
│   ├── inspect_activations.py
│   ├── random_read.py
│   ├── intervention.py
│   ├── layer_sweep.py
│   └── memory_test.py
│
├── analysis/
│   └── plots.py
│
└── results/
```

`model.py`

```text
load Pythia
tokenization
forward passes
```

`relay.py`

```text
read state
update state
write state
```

`experiments/`

Keep individual experiments isolated rather than turning `main.py` into a 1,000-line research notebook.

---

# The roadmap I would actually follow

Don't think about all ten phases at once. Your next progression should be:

**1. Forward pass** ✅  
**2. Extract `[768]` hidden vector** ← you're here  
**3. Compute scalar \(z=u^\top h\)**  
**4. Persist \(z\) between calls**  
**5. Inject \(\alpha zv\) into another forward pass**  
**6. Show changing \(z\) changes output**  
**7. Sweep layers / intervention strength**  
**8. Give \(z\) an interpretable meaning**  
**9. Demonstrate cross-context latent memory**  
**10. Expand to multidimensional/trainable latent state**

The really important milestone isn't "build simulated consciousness." It's much narrower:

> **Demonstrate that a frozen transformer can read information into a tiny persistent latent channel and later have its behavior causally altered by information carried through that channel.**

If you can demonstrate that cleanly, you already have a legitimately interesting ML systems / interpretability project.

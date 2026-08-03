"""Small statistics helpers (numpy only): bootstrap intervals, permutation tests, AUROC."""

import numpy as np


def bootstrap_ci(values, *, level=0.95, resamples=10000, seed=0):
    """Mean and percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("need at least two values")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, (resamples, values.size))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(values.mean()), float(low), float(high)


def sign_flip_p(values, *, resamples=10000, seed=0):
    """Two-sided p-value that the mean is zero, by randomly flipping signs."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(resamples, values.size))
    observed = abs(values.mean())
    null = np.abs((flips * values).mean(axis=1))
    return float((1 + (null >= observed - 1e-12).sum()) / (1 + resamples))


def auroc(scores, labels):
    """Probability a random positive outscores a random negative (ties count half)."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], scores[~labels]
    if not positive.size or not negative.size:
        raise ValueError("need both classes")
    greater = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(greater + 0.5 * ties)


def spearman(x, y):
    """Rank correlation (no tie correction needed for the small grids used here)."""
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float)))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])

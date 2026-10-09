"""Uncertainty for error rates.

A single WER number on a few hundred utterances hides a wide interval. Gyara
reports every corpus WER with a bootstrap 95% confidence interval, and every
"A is better than B" claim with a paired bootstrap test (Bisani & Ney, 2004).

Resampling is done over *clusters*, not utterances, whenever cluster ids are
available. Utterances from the same speaker (or, in FLEURS, the same sentence
read by several speakers) are correlated; resampling them independently makes
intervals look tighter than they are.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

DEFAULT_B = 10_000
DEFAULT_SEED = 1234


def _clustered(errors: Sequence[float], lengths: Sequence[float], groups: Sequence | None):
    e = np.asarray(errors, dtype=float)
    n = np.asarray(lengths, dtype=float)
    if groups is None or all(g is None for g in groups):
        return e, n, len(e)
    keys = [g if g is not None else f"__solo_{i}" for i, g in enumerate(groups)]
    uniq, inv = np.unique(np.asarray(keys, dtype=object).astype(str), return_inverse=True)
    ec = np.bincount(inv, weights=e, minlength=len(uniq))
    nc = np.bincount(inv, weights=n, minlength=len(uniq))
    return ec, nc, len(uniq)


@dataclass
class Interval:
    estimate: float
    low: float
    high: float
    n_units: int
    unit: str
    resamples: int

    def to_dict(self) -> dict:
        return asdict(self)

    def fmt(self, pct: bool = True) -> str:
        k = 100 if pct else 1
        return f"{self.estimate * k:.2f} [{self.low * k:.2f}, {self.high * k:.2f}]"


def bootstrap_rate(
    errors: Sequence[float],
    lengths: Sequence[float],
    groups: Sequence | None = None,
    b: int = DEFAULT_B,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Percentile bootstrap CI for a pooled rate sum(errors)/sum(lengths)."""
    e, n, k = _clustered(errors, lengths, groups)
    total = n.sum()
    est = e.sum() / total if total else float("nan")
    if k < 2 or not total:
        return Interval(est, float("nan"), float("nan"), k, _unit(groups), 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(b, k))
    denom = n[idx].sum(axis=1)
    rates = np.divide(e[idx].sum(axis=1), denom, out=np.full(b, np.nan), where=denom > 0)
    lo, hi = np.nanquantile(rates, [alpha / 2, 1 - alpha / 2])
    return Interval(float(est), float(lo), float(hi), k, _unit(groups), b)


@dataclass
class PairedResult:
    """System A minus system B. Negative delta means A has fewer errors."""

    rate_a: float
    rate_b: float
    delta: float
    low: float
    high: float
    p_value: float
    a_better_prob: float
    n_units: int
    unit: str
    resamples: int
    mde: float  # minimum detectable effect at 80% power, alpha 0.05 (two-sided)

    @property
    def significant(self) -> bool:
        return not (self.low <= 0.0 <= self.high)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d

    def sentence(self, a: str = "A", b: str = "B") -> str:
        """Plain-English verdict, written so it cannot over-claim."""
        pp = lambda x: f"{abs(x) * 100:.2f}"  # noqa: E731
        if math.isnan(self.delta):
            return "Not enough data to compare."
        if not self.significant:
            return (
                f"No reliable difference: {a} vs {b} differ by {pp(self.delta)} points, "
                f"inside the noise (95% CI {self.low * 100:+.2f} to {self.high * 100:+.2f}). "
                f"With this test set, differences under about {self.mde * 100:.2f} points "
                f"cannot be detected."
            )
        better, worse = (a, b) if self.delta < 0 else (b, a)
        return (
            f"{better} makes fewer errors than {worse}: {pp(self.delta)} points "
            f"(95% CI {self.low * 100:+.2f} to {self.high * 100:+.2f}, p={self.p_value:.4f})."
        )


def paired_bootstrap(
    errors_a: Sequence[float],
    errors_b: Sequence[float],
    lengths: Sequence[float],
    groups: Sequence | None = None,
    b: int = DEFAULT_B,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> PairedResult:
    """Paired bootstrap on the same utterances scored by two systems.

    ``lengths`` are reference lengths, identical for both systems by
    construction (same references, same normaliser).
    """
    if not (len(errors_a) == len(errors_b) == len(lengths)):
        raise ValueError("paired inputs must be aligned and the same length")
    ea, n, k = _clustered(errors_a, lengths, groups)
    eb, _, _ = _clustered(errors_b, lengths, groups)
    total = n.sum()
    if not total or k < 2:
        nan = float("nan")
        return PairedResult(nan, nan, nan, nan, nan, nan, nan, k, _unit(groups), 0, nan)
    ra, rb = ea.sum() / total, eb.sum() / total
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(b, k))
    denom = n[idx].sum(axis=1)
    ok = denom > 0
    d = (ea[idx].sum(axis=1) - eb[idx].sum(axis=1))[ok] / denom[ok]
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    # Two-sided p-value: how often the resampled delta lands on the other side
    # of zero from the observed one (or on it).
    obs = ra - rb
    if obs < 0:
        p = 2 * np.mean(d >= 0)
    elif obs > 0:
        p = 2 * np.mean(d <= 0)
    else:
        p = 1.0
    se = float(np.std(d, ddof=1))
    return PairedResult(
        rate_a=float(ra),
        rate_b=float(rb),
        delta=float(obs),
        low=float(lo),
        high=float(hi),
        p_value=float(min(1.0, p)),
        a_better_prob=float(np.mean(d < 0)),
        n_units=k,
        unit=_unit(groups),
        resamples=int(ok.sum()),
        mde=2.8 * se,  # (z_0.975 + z_0.80) * SE
    )


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation, ignoring pairs where either value is NaN."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[m], ya[m]
    if len(xa) < 3:
        return float("nan")
    rx = _rank(xa)
    ry = _rank(ya)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    # Average ranks over ties.
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def _unit(groups) -> str:
    if groups is None or all(g is None for g in groups):
        return "utterance"
    return "cluster"

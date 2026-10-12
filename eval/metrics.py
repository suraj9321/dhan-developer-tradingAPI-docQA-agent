"""Retrieval, answer, refusal, latency and cost metrics.

Retrieval is scored at section level (see gen_questions for why), on the
de-duplicated list of sections in rank order, so a retriever is not
rewarded for returning three chunks of the same section.
"""

from __future__ import annotations

import math


def dedupe(seq: list[str]) -> list[str]:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Strict: fraction of the question's source sections found in the top-k."""
    if not gold:
        return 0.0
    return len(set(dedupe(retrieved)[:k]) & gold) / len(gold)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Pooled: 1 if any section judged to answer the question is in the top-k."""
    return float(bool(set(dedupe(retrieved)[:k]) & relevant))


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, s in enumerate(dedupe(retrieved), 1):
        if s in relevant:
            return 1.0 / i
    return 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    idx = (len(xs) - 1) * p / 100
    lo, hi = math.floor(idx), math.ceil(idx)
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)


def mean(values) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)

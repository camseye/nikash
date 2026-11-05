"""Statistics used in the paper (G2 design §4), standard library only."""

from __future__ import annotations

import math
import random
from typing import Callable, Sequence


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def design_effect(clusters: Sequence[tuple[float, float]]) -> float:
    """Kish design effect of a proportion pooled over clusters of (successes, trials).

    Cases from one cluster (one attack step) are correlated and carry less
    information than independent ones; n divided by this is the effective
    sample size. When the outcomes don't vary (all successes or all
    failures) it can't be estimated, and each cluster counts as one case.
    """
    clusters = [(k, n) for k, n in clusters if n]
    total = sum(n for _, n in clusters)
    count = len(clusters)
    if total == 0 or count == total:
        return 1.0
    p = sum(k for k, _ in clusters) / total
    if count < 2 or p in (0.0, 1.0):
        return total / count
    between = count / (count - 1) * sum((k - p * n) ** 2 for k, n in clusters) / total**2
    return max(1.0, between / (p * (1 - p) / total))


MIN_OUTCOMES = 5


def clustered_wilson(clusters: Sequence[tuple[float, float]], z: float = 1.96) -> tuple[float, float, float]:
    """Wilson interval on the effective sample size: (low, high, n_eff).

    With fewer than MIN_OUTCOMES successes or failures the design effect
    can't be estimated reliably (one differing outcome would make it about
    1), so each cluster counts as one case. Otherwise n_eff is n over the
    design effect, but never fewer than the number of clusters. Unclustered
    data (one case per cluster) keeps n_eff = n.
    """
    clusters = [(k, n) for k, n in clusters if n]
    total = sum(n for _, n in clusters)
    if total == 0:
        return (0.0, 1.0, 0.0)
    k = sum(k for k, _ in clusters)
    count = len(clusters)
    if count == total:
        n_eff = float(total)
    elif min(k, total - k) < MIN_OUTCOMES:
        n_eff = float(count)
    else:
        n_eff = max(float(count), total / design_effect(clusters))
    lo, hi = wilson(k / total * n_eff, n_eff, z)
    return (lo, hi, n_eff)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's tau-b (handles ties) over the positions where neither value
    is nan. Returns nan when undefined, including when fewer than 2 remain."""
    if len(x) != len(y):
        raise ValueError("sequences differ in length")
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 2:
        return float("nan")
    x, y = [a for a, _ in pairs], [b for _, b in pairs]
    concordant = discordant = ties_x = ties_y = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denom if denom else float("nan")


def bootstrap(statistic: Callable[[random.Random], float], n_boot: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> tuple[float, float, int]:
    """Percentile interval and the number of resamples whose statistic was
    undefined (nan) and so left out: (low, high, undefined).
    `statistic(rng)` must draw its own resample from rng."""
    rng = random.Random(seed)
    drawn = [statistic(rng) for _ in range(n_boot)]
    values = sorted(v for v in drawn if not math.isnan(v))
    undefined = len(drawn) - len(values)
    if not values:
        return (float("nan"), float("nan"), undefined)
    lo = values[int((alpha / 2) * (len(values) - 1))]
    hi = values[int((1 - alpha / 2) * (len(values) - 1))]
    return (lo, hi, undefined)


def bootstrap_ci(statistic: Callable[[random.Random], float], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval over the resamples where the statistic is defined; see `bootstrap`."""
    return bootstrap(statistic, n_boot, seed, alpha)[:2]

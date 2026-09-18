"""Effect sizes and confidence intervals.

A p-value says the effect is probably not zero, not whether it matters. So:
- Hedges' g = Cohen's d with the small sample correction J = 1 - 3/(4df - 1)
  (d overestimates by a few percent below n ~ 50 per arm)
- Cliff's delta next to it, ordinal, so meaningful for bounded scores
- BCa bootstrap intervals instead of percentile ones, the acceleration term
  corrects for skew which bounded learning gains always have
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from .special import normal_cdf, normal_ppf
from .tests import mean, stdev, variance


@dataclass(frozen=True)
class EffectSize:
    name: str
    value: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    interpretation: str = ""

    def __str__(self) -> str:
        if self.ci_low is None:
            return f"{self.name} = {self.value:.3f}"
        return f"{self.name} = {self.value:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}]"


def interpret_d(d: float) -> str:
    """Cohen's bands. Rules of thumb, not thresholds."""
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised mean difference with the pooled sd."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("Cohen's d needs at least two observations per group")
    pooled = math.sqrt(((na - 1) * variance(a) + (nb - 1) * variance(b)) / (na + nb - 2))
    if pooled <= 0.0:
        return 0.0
    return (mean(a) - mean(b)) / pooled


def hedges_g(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Cohen's d with the small sample correction."""
    d = cohens_d(a, b)
    df = len(a) + len(b) - 2
    correction = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 1 else 1.0
    g = d * correction
    return EffectSize("Hedges g", g, interpretation=interpret_d(g))


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Ordinal effect size: P(a > b) - P(a < b), in [-1, 1]."""
    if not a or not b:
        raise ValueError("Cliff's delta needs non-empty samples")
    # sorting turns the naive O(n*m) into O(n log n + m log m)
    ordered = sorted(b)
    greater = lesser = 0
    for value in a:
        low, high = _bounds(ordered, value)
        greater += low
        lesser += len(ordered) - high
    total = len(a) * len(b)
    delta = (greater - lesser) / total
    magnitude = abs(delta)
    label = (
        "negligible" if magnitude < 0.147
        else "small" if magnitude < 0.33
        else "medium" if magnitude < 0.474
        else "large"
    )
    return EffectSize("Cliff's delta", delta, interpretation=label)


def _bounds(ordered: Sequence[float], value: float) -> Tuple[int, int]:
    """(count strictly below, index one past the last equal)."""
    low, high = 0, len(ordered)
    while low < high:
        mid = (low + high) // 2
        if ordered[mid] < value:
            low = mid + 1
        else:
            high = mid
    left = low
    low, high = left, len(ordered)
    while low < high:
        mid = (low + high) // 2
        if ordered[mid] <= value:
            low = mid + 1
        else:
            high = mid
    return left, low


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Effect size that goes with Mann-Whitney U (equals Cliff's delta)."""
    delta = cliffs_delta(a, b)
    return EffectSize("rank-biserial r", delta.value, interpretation=delta.interpretation)


# bootstrap
def bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    statistic: Callable[[Sequence[float], Sequence[float]], float],
    *,
    resamples: int = 4000,
    confidence: float = 0.95,
    rng: Optional[random.Random] = None,
) -> Tuple[float, float, float]:
    """BCa bootstrap interval, returns (observed, low, high).

    - z0 (bias correction) comes from the share of bootstrap replicates below
      the observed value, fixes a bootstrap distribution not centred on the
      estimate
    - acceleration is estimated by jackknife, fixes a standard error that
      changes with the parameter (it does for standardised differences since
      the denominator is estimated too)

    A percentile interval would miscover exactly in this regime.
    """
    rng = rng or random.Random(20260909)
    observed = statistic(a, b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return observed, observed, observed

    replicates = []
    for _ in range(resamples):
        sample_a = [a[rng.randrange(na)] for _ in range(na)]
        sample_b = [b[rng.randrange(nb)] for _ in range(nb)]
        try:
            replicates.append(statistic(sample_a, sample_b))
        except (ValueError, ZeroDivisionError):
            continue
    if len(replicates) < 20:
        return observed, observed, observed
    replicates.sort()

    below = sum(1 for value in replicates if value < observed)
    proportion = below / len(replicates)
    proportion = min(max(proportion, 1e-6), 1 - 1e-6)
    z0 = normal_ppf(proportion)

    # jackknife over the pooled observations for the acceleration term
    jackknife = []
    for index in range(na):
        reduced = a[:index] + a[index + 1 :]
        if len(reduced) >= 2:
            jackknife.append(statistic(reduced, b))
    for index in range(nb):
        reduced = b[:index] + b[index + 1 :]
        if len(reduced) >= 2:
            jackknife.append(statistic(a, reduced))
    if len(jackknife) < 3:
        acceleration = 0.0
    else:
        jack_mean = mean(jackknife)
        numerator = math.fsum((jack_mean - value) ** 3 for value in jackknife)
        denominator = 6.0 * (math.fsum((jack_mean - value) ** 2 for value in jackknife) ** 1.5)
        acceleration = numerator / denominator if denominator > 1e-12 else 0.0

    alpha = (1.0 - confidence) / 2.0
    def adjusted(z_alpha: float) -> float:
        numerator = z0 + z_alpha
        denominator = 1.0 - acceleration * numerator
        if abs(denominator) < 1e-12:
            return 0.5
        return normal_cdf(z0 + numerator / denominator)

    low_q = adjusted(normal_ppf(alpha))
    high_q = adjusted(normal_ppf(1.0 - alpha))
    low_index = min(len(replicates) - 1, max(0, int(low_q * len(replicates))))
    high_index = min(len(replicates) - 1, max(0, int(high_q * len(replicates))))
    if low_index > high_index:
        low_index, high_index = high_index, low_index
    return observed, replicates[low_index], replicates[high_index]


def effect_with_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int = 4000,
    confidence: float = 0.95,
    rng: Optional[random.Random] = None,
) -> EffectSize:
    """Hedges' g with a BCa bootstrap CI."""
    def statistic(x: Sequence[float], y: Sequence[float]) -> float:
        return hedges_g(x, y).value

    value, low, high = bootstrap_ci(
        list(a), list(b), statistic, resamples=resamples, confidence=confidence, rng=rng
    )
    return EffectSize("Hedges g", value, low, high, interpret_d(value))

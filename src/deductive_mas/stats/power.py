"""Power and sample size planning.
The protocol has to state the detectable effect size before collecting data,
otherwise a null result means nothing. Uses the normal shift approximation
to the non central t, good to a few thousandths at these sample sizes.
"""

import math
from dataclasses import dataclass
from typing import Optional

from .special import normal_cdf, normal_ppf, t_ppf


@dataclass(frozen=True)
class PowerAnalysis:
    effect_size: float
    n_per_group: int
    alpha: float
    power: float

    def __str__(self) -> str:
        return (
            f"d = {self.effect_size:.2f}, n = {self.n_per_group}/group, "
            f"alpha = {self.alpha:.3f}, power = {self.power:.3f}"
        )


def two_sample_power(effect_size: float, n_per_group: int, alpha: float = 0.05) -> float:
    """Power of a two sided two sample t-test."""
    if n_per_group < 2:
        return 0.0
    df = 2 * n_per_group - 2
    critical = t_ppf(1.0 - alpha / 2.0, df)
    ncp = abs(effect_size) * math.sqrt(n_per_group / 2.0)
    # normal approx to the non central t, ~1e-3 accurate for df >= 20
    return normal_cdf(ncp - critical) + normal_cdf(-ncp - critical)


def required_sample_size(
    effect_size: float, power: float = 0.80, alpha: float = 0.05, *, maximum: int = 100_000
) -> int:
    """Smallest per group n reaching power, by bisection."""
    if effect_size == 0.0:
        return maximum
    low, high = 2, 64
    while high < maximum and two_sample_power(effect_size, high, alpha) < power:
        low, high = high, high * 2
    if two_sample_power(effect_size, high, alpha) < power:
        return maximum
    while low < high:
        mid = (low + high) // 2
        if two_sample_power(effect_size, mid, alpha) < power:
            low = mid + 1
        else:
            high = mid
    return low


def minimum_detectable_effect(n_per_group: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest effect this design can detect, by bisection on power."""
    low, high = 0.0, 5.0
    for _ in range(80):
        mid = 0.5 * (low + high)
        if two_sample_power(mid, n_per_group, alpha) < power:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def plan(effect_size: float, power: float = 0.80, alpha: float = 0.05) -> PowerAnalysis:
    n = required_sample_size(effect_size, power, alpha)
    return PowerAnalysis(effect_size, n, alpha, two_sample_power(effect_size, n, alpha))

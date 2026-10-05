"""Multiple comparison corrections.

Several RQs x several outcomes, so alpha = .05 uncorrected would give a
false positive well above 5%. Two corrections for two questions:
- Holm-Bonferroni controls the family wise error rate (any false positive),
  for the confirmatory hypotheses
- Benjamini-Hochberg controls the false discovery rate, for the exploratory
  sweep over misconception categories where power matters more
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Adjusted:
    label: str
    raw: float
    adjusted: float
    rejected: bool


def holm_bonferroni(p_values: Sequence[Tuple[str, float]], alpha: float = 0.05) -> List[Adjusted]:
    """Step down FWER control.
    Sort ascending, compare the k-th smallest against alpha / (m - k), stop at
    the first failure. Uniformly more powerful than plain Bonferroni.
    """
    ordered = sorted(p_values, key=lambda kv: kv[1])
    m = len(ordered)
    out: List[Adjusted] = []
    running = 0.0
    still_rejecting = True
    for index, (label, p) in enumerate(ordered):
        adjusted = min(1.0, (m - index) * p)
        running = max(running, adjusted)          # keep monotone
        if still_rejecting and running > alpha:
            still_rejecting = False
        out.append(Adjusted(label, p, running, still_rejecting))
    return out


def benjamini_hochberg(p_values: Sequence[Tuple[str, float]], alpha: float = 0.05) -> List[Adjusted]:
    """Step up FDR control (independent or PRDS statistics)."""
    ordered = sorted(p_values, key=lambda kv: kv[1])
    m = len(ordered)
    if m == 0:
        return []
    adjusted: List[float] = [0.0] * m
    previous = 1.0
    # walk from the largest p down so the adjusted values stay monotone
    for index in range(m - 1, -1, -1):
        value = min(1.0, ordered[index][1] * m / (index + 1))
        previous = min(previous, value)
        adjusted[index] = previous
    threshold_index = -1
    for index in range(m - 1, -1, -1):
        if ordered[index][1] <= alpha * (index + 1) / m:
            threshold_index = index
            break
    return [
        Adjusted(label, p, adjusted[index], index <= threshold_index)
        for index, (label, p) in enumerate(ordered)
    ]


def summarise(results: Sequence[Adjusted]) -> str:
    rejected = sum(1 for r in results if r.rejected)
    return f"{rejected}/{len(results)} hypotheses rejected after correction"

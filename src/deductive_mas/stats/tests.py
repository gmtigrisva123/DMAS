"""Hypothesis tests.

t-tests plus the non parametric alternatives, because learning gains are
bounded, discrete and usually skewed. Both get reported (reporting only the
significant one would be the classic forking paths mistake). Welch is the
default instead of Student's pooled version since the arms have no reason to
share a variance.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .special import chi2_sf, normal_cdf, t_cdf, t_ppf, t_sf


@dataclass(frozen=True)
class TestResult:
    """Outcome of one test with everything needed to report it."""

    name: str
    statistic: float
    p_value: float
    df: Optional[float] = None
    alternative: str = "two-sided"
    detail: str = ""

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def apa(self) -> str:
        """Citation ready string, e.g. t(37.2) = 3.41, p = .0015."""
        p = self.p_value
        rendered = "p < .001" if p < 0.001 else f"p = {p:.4f}".replace("0.", ".", 1)
        if self.df is None:
            return f"{self.name} = {self.statistic:.3f}, {rendered}"
        return f"{self.name}({self.df:.1f}) = {self.statistic:.3f}, {rendered}"


def mean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("mean of an empty sample")
    return math.fsum(xs) / len(xs)


def variance(xs: Sequence[float], *, ddof: int = 1) -> float:
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    mu = mean(xs)
    return math.fsum((x - mu) ** 2 for x in xs) / (n - ddof)


def stdev(xs: Sequence[float], *, ddof: int = 1) -> float:
    return math.sqrt(variance(xs, ddof=ddof))


# parametric
def welch_t_test(a: Sequence[float], b: Sequence[float], *, alternative: str = "two-sided") -> TestResult:
    """Welch's t-test with Welch-Satterthwaite df."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("Welch's t-test needs at least two observations per group")
    va, vb = variance(a), variance(b)
    sa, sb = va / na, vb / nb
    denominator = math.sqrt(sa + sb)
    if denominator <= 0.0:
        return TestResult("t", 0.0, 1.0, float(na + nb - 2), alternative, "zero variance in both groups")
    t = (mean(a) - mean(b)) / denominator
    df = (sa + sb) ** 2 / ((sa * sa) / (na - 1) + (sb * sb) / (nb - 1))
    return TestResult("t", t, _t_p_value(t, df, alternative), df, alternative)


def student_t_test(a: Sequence[float], b: Sequence[float], *, alternative: str = "two-sided") -> TestResult:
    """Pooled variance t-test (only for comparison with Welch)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("the t-test needs at least two observations per group")
    df = na + nb - 2
    pooled = ((na - 1) * variance(a) + (nb - 1) * variance(b)) / df
    denominator = math.sqrt(pooled * (1.0 / na + 1.0 / nb))
    if denominator <= 0.0:
        return TestResult("t", 0.0, 1.0, float(df), alternative, "zero pooled variance")
    t = (mean(a) - mean(b)) / denominator
    return TestResult("t", t, _t_p_value(t, df, alternative), float(df), alternative)


def paired_t_test(a: Sequence[float], b: Sequence[float], *, alternative: str = "two-sided") -> TestResult:
    """Paired t-test on the differences (pre/post)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    differences = [x - y for x, y in zip(a, b)]
    n = len(differences)
    if n < 2:
        raise ValueError("the paired t-test needs at least two pairs")
    sd = stdev(differences)
    if sd <= 0.0:
        return TestResult("t", 0.0, 1.0, float(n - 1), alternative, "all differences identical")
    t = mean(differences) / (sd / math.sqrt(n))
    return TestResult("t", t, _t_p_value(t, n - 1, alternative), float(n - 1), alternative)


def _t_p_value(t: float, df: float, alternative: str) -> float:
    if alternative == "greater":
        return t_sf(t, df)
    if alternative == "less":
        return t_cdf(t, df)
    if alternative != "two-sided":
        raise ValueError(f"unknown alternative: {alternative!r}")
    return 2.0 * min(t_cdf(t, df), t_sf(t, df))


# non parametric
def rank(values: Sequence[float]) -> Tuple[list, float]:
    """Mid ranks plus the tie correction term for rank tests."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_term = 0.0
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = average
        group = stop - index + 1
        if group > 1:
            tie_term += group ** 3 - group
        index = stop + 1
    return ranks, tie_term


def mann_whitney_u(a: Sequence[float], b: Sequence[float], *, alternative: str = "two-sided") -> TestResult:
    """Mann-Whitney U with tie corrected normal approximation.
    Normal approximation because the arms are big (n >= 20 per group by
    design). For small samples treat the p-value as indicative only.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        raise ValueError("Mann-Whitney U needs non-empty samples")
    combined = list(a) + list(b)
    ranks, tie_term = rank(combined)
    rank_sum_a = math.fsum(ranks[:na])
    u_a = rank_sum_a - na * (na + 1) / 2.0
    u_b = na * nb - u_a
    u = min(u_a, u_b)

    n = na + nb
    mu = na * nb / 2.0
    sigma_squared = (na * nb / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if sigma_squared <= 0.0:
        return TestResult("U", u, 1.0, None, alternative, "degenerate variance")
    # continuity correction of 0.5
    z = (u_a - mu) / math.sqrt(sigma_squared)
    if alternative == "two-sided":
        p = 2.0 * (1.0 - normal_cdf(abs(z)))
    elif alternative == "greater":
        p = 1.0 - normal_cdf(z)
    elif alternative == "less":
        p = normal_cdf(z)
    else:
        raise ValueError(f"unknown alternative: {alternative!r}")
    return TestResult("U", u, min(1.0, max(0.0, p)), None, alternative, f"z = {z:.3f}")


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float], *, alternative: str = "two-sided") -> TestResult:
    """Paired non parametric test on the signed ranks of the differences."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    differences = [x - y for x, y in zip(a, b) if x != y]
    n = len(differences)
    if n == 0:
        return TestResult("W", 0.0, 1.0, None, alternative, "all pairs tied")
    ranks, tie_term = rank([abs(d) for d in differences])
    positive = math.fsum(r for r, d in zip(ranks, differences) if d > 0)
    negative = math.fsum(r for r, d in zip(ranks, differences) if d < 0)
    w = min(positive, negative)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt((n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0)
    if sigma <= 0.0:
        return TestResult("W", w, 1.0, None, alternative, "degenerate variance")
    z = (positive - mu) / sigma
    if alternative == "two-sided":
        p = 2.0 * (1.0 - normal_cdf(abs(z)))
    elif alternative == "greater":
        p = 1.0 - normal_cdf(z)
    else:
        p = normal_cdf(z)
    return TestResult("W", w, min(1.0, max(0.0, p)), None, alternative, f"z = {z:.3f}")


def chi_square_independence(table: Sequence[Sequence[float]]) -> TestResult:
    """Pearson chi squared on a contingency table."""
    rows = len(table)
    cols = len(table[0]) if rows else 0
    if rows < 2 or cols < 2:
        raise ValueError("the chi-squared test needs at least a 2x2 table")
    total = math.fsum(math.fsum(row) for row in table)
    if total <= 0:
        raise ValueError("contingency table is empty")
    row_totals = [math.fsum(row) for row in table]
    col_totals = [math.fsum(table[r][c] for r in range(rows)) for c in range(cols)]
    statistic = 0.0
    small = 0
    for r in range(rows):
        for c in range(cols):
            expected = row_totals[r] * col_totals[c] / total
            if expected <= 0:
                continue
            if expected < 5:
                small += 1
            statistic += (table[r][c] - expected) ** 2 / expected
    df = (rows - 1) * (cols - 1)
    detail = f"{small} cell(s) with expected count below 5" if small else ""
    return TestResult("chi2", statistic, chi2_sf(statistic, df), float(df), "two-sided", detail)

"""Special functions behind the tests.

Everything comes from two things: the regularised incomplete beta (Student
t, F, binomial tail) and the regularised incomplete gamma (chi squared,
Poisson). Both via the continued fraction expansions from Numerical Recipes
with Lentz's algorithm. Written here instead of imported so every p-value
can be traced to code in this repo.
"""

import math
from typing import Callable

_EPS = 3.0e-16
_FPMIN = 1.0e-300
_MAXIT = 300


# beta family
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the CDF of Beta(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    # the expansion only converges fast on one side of the mode, reflect otherwise
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


# gamma family
def _gser(a: float, x: float) -> float:
    """Series for the regularised lower incomplete gamma."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_MAXIT):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Continued fraction for the regularised upper incomplete gamma."""
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAXIT + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def regularised_lower_gamma(a: float, x: float) -> float:
    """P(a, x), the CDF of Gamma(a, 1)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("regularised_lower_gamma requires x >= 0 and a > 0")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


# distributions
def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation + one Halley step).
    The rational approximation is ~1e-9, the Halley step with the exact CDF
    takes it to machine precision, needed because the power calculations
    invert this in the tails.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("normal_ppf requires 0 < p < 1")
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    elif p > phigh:
        q = math.sqrt(-2.0 * math.log1p(-p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    else:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
    error = normal_cdf(x) - p
    density = normal_pdf(x)
    if density > 1e-300:
        u = error / density
        x -= u / (1.0 + 0.5 * x * u)          # Halley step
    return x


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t with df degrees of freedom."""
    if df <= 0:
        raise ValueError("t_cdf requires df > 0")
    x = df / (df + t * t)
    tail = 0.5 * regularised_incomplete_beta(0.5 * df, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_sf(t: float, df: float) -> float:
    """Upper tail of Student's t."""
    return 1.0 - t_cdf(t, df)


def t_ppf(p: float, df: float) -> float:
    """Inverse t CDF by bisection (the CDF is monotone)."""
    if not 0.0 < p < 1.0:
        raise ValueError("t_ppf requires 0 < p < 1")
    lo, hi = -400.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def chi2_cdf(x: float, df: float) -> float:
    if x <= 0.0:
        return 0.0
    return regularised_lower_gamma(0.5 * df, 0.5 * x)


def chi2_sf(x: float, df: float) -> float:
    return 1.0 - chi2_cdf(x, df)


def f_sf(f: float, df1: float, df2: float) -> float:
    """Upper tail of F."""
    if f <= 0.0:
        return 1.0
    return regularised_incomplete_beta(0.5 * df2, 0.5 * df1, df2 / (df2 + df1 * f))

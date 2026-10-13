"""The few special functions and exact tests the statistics need, in plain Python (no SciPy).

Regularised incomplete gamma by series / continued fraction (Numerical Recipes style); chi-square, gamma and Poisson
distributions follow from it. Accuracy is ~1e-12 over the ranges used here and is checked against closed forms in the tests.
"""
from __future__ import annotations

import math

import numpy as np


def _gamma_series(a, x):
    term = total = 1.0 / a
    n = a
    for _ in range(10_000):
        n += 1.0
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a, x):
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammainc(a, x):
    """Regularised lower incomplete gamma P(a, x)."""
    if a <= 0:
        raise ValueError("a must be positive")
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        return min(1.0, _gamma_series(a, x))
    return max(0.0, 1.0 - _gamma_cf(a, x))


def gammaincc(a, x):
    """Regularised upper incomplete gamma Q(a, x)."""
    if x <= 0:
        return 1.0
    if x < a + 1.0:
        return max(0.0, 1.0 - _gamma_series(a, x))
    return min(1.0, _gamma_cf(a, x))


def chi2_sf(x, k):
    return gammaincc(k / 2.0, x / 2.0)


def chi2_cdf(x, k):
    return gammainc(k / 2.0, x / 2.0)


def gamma_ppf(q, a):
    """Quantile of Gamma(shape a, scale 1) by bisection on P(a, x) = q."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between 0 and 1")
    lo, hi = 0.0, a + 12.0 * math.sqrt(a) + 40.0
    while gammainc(a, hi) < q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if gammainc(a, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_sf(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def norm_ppf(q):
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between 0 and 1")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def poisson_cdf(k, mu):
    """P(X <= k) for X ~ Poisson(mu), exact (through the upper incomplete gamma)."""
    if k < 0:
        return 0.0
    if mu <= 0:
        return 1.0
    return gammaincc(k + 1.0, mu)


def poisson_sf(k, mu):
    """P(X > k)."""
    return 1.0 - poisson_cdf(k, mu) if mu > 0 else 0.0


def poisson_two_sided_p(k, mu):
    """Two-sided exact Poisson p-value: twice the smaller tail, capped at 1."""
    lower = poisson_cdf(k, mu)
    upper = 1.0 - poisson_cdf(k - 1, mu) if k > 0 else 1.0
    return min(1.0, 2.0 * min(lower, upper))


def poisson_interval(k, level=0.95):
    """Garwood exact confidence interval for a Poisson mean given one count ``k``."""
    alpha = 1.0 - level
    lo = 0.0 if k == 0 else gamma_ppf(alpha / 2.0, float(k))
    hi = gamma_ppf(1.0 - alpha / 2.0, float(k + 1))
    return lo, hi


def binom_pmf_array(n, p):
    k = np.arange(n + 1)
    logc = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
    with np.errstate(divide="ignore"):
        lp = logc + k * math.log(p) + (n - k) * math.log1p(-p) if 0.0 < p < 1.0 else np.where(k == (n if p >= 1.0 else 0), 0.0, -np.inf)
    return np.exp(lp)


def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(min(1.0, binom_pmf_array(n, p)[: k + 1].sum()))


def binom_two_sided_p(k, n, p):
    """Exact two-sided binomial p-value (sum of probabilities not larger than that of the observed count)."""
    pmf = binom_pmf_array(n, p)
    return float(min(1.0, pmf[pmf <= pmf[k] * (1 + 1e-9)].sum()))


def ks_uniform(sample):
    """Kolmogorov-Smirnov distance and asymptotic p-value of ``sample`` against the uniform distribution on [0, 1]."""
    x = np.sort(np.asarray(sample, float))
    n = len(x)
    if n == 0:
        raise ValueError("empty sample")
    i = np.arange(1, n + 1)
    d = float(max((i / n - x).max(), (x - (i - 1) / n).max()))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    if lam < 0.2:
        return d, 1.0
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101))
    return d, float(min(1.0, max(0.0, p)))


def betainc(x, a, b):
    """Regularised incomplete beta I_x(a, b) by continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))

    def cf(a_, b_, x_):
        tiny = 1e-300
        qab, qap, qam = a_ + b_, a_ + 1.0, a_ - 1.0
        c, d = 1.0, 1.0 - qab * x_ / qap
        d = 1.0 / (tiny if abs(d) < tiny else d)
        h = d
        for m_ in range(1, 500):
            m2 = 2 * m_
            aa = m_ * (b_ - m_) * x_ / ((qam + m2) * (a_ + m2))
            d = 1.0 + aa * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
            aa = -(a_ + m_) * (qab + m_) * x_ / ((a_ + m2) * (qap + m2))
            d = 1.0 + aa * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-14:
                break
        return h

    if x < (a + 1.0) / (a + b + 2.0):
        return front * cf(a, b, x) / a
    return 1.0 - front * cf(b, a, 1.0 - x) / b


def beta_ppf(q, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if betainc(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k, n, level=0.95):
    """Exact binomial confidence interval for ``k`` successes in ``n`` trials."""
    alpha = 1.0 - level
    lo = 0.0 if k == 0 else beta_ppf(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else beta_ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return lo, hi


def t_cdf(x, df):
    """Student t cumulative distribution function."""
    if df <= 0:
        raise ValueError("df must be positive")
    z = df / (df + x * x)
    tail = 0.5 * betainc(z, df / 2.0, 0.5)
    return 1.0 - tail if x > 0 else tail


def t_sf(x, df):
    return 1.0 - t_cdf(x, df)


def t_ppf(q, df):
    """Quantile of the Student t distribution by bisection."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly between 0 and 1")
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def normal_two_sided_p(z):
    return 2.0 * norm_sf(abs(z))

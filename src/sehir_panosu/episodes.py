"""Regional pollution episodes: when many stations are high at once, and whether a model that knows about dust agrees.

A city-wide PM10 episode shows up at every station together (a regional signal such as Saharan dust), while NO2 peaks are local; the share of
variance in the first principal component of the stations' log daily values measures that.
"""
from __future__ import annotations

import math

import numpy as np


def regional_median(stack, min_stations=3):
    """Median over stations (rows) for every day (columns); days with fewer than ``min_stations`` valid stations are NaN."""
    stack = np.asarray(stack, float)
    cnt = np.isfinite(stack).sum(axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
    med[cnt < min_stations] = np.nan
    return med, cnt


def find_episodes(dates, regional, threshold, min_days=1, max_gap=0):
    """Runs of days at or above ``threshold``; returns dicts ``start, end, days, peak, mean``. Runs separated by at most ``max_gap`` days merge."""
    ok = np.isfinite(regional) & (regional >= threshold)
    runs, i = [], 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if b - a + 1 >= min_days:
            seg = regional[a:b + 1]
            out.append({"start": dates[a], "end": dates[b], "days": int(b - a + 1), "peak": float(np.nanmax(seg)), "mean": float(np.nanmean(seg))})
    return out


def first_component_share(stack, min_days=200):
    """Share of the variance of the stations' standardised log daily values carried by the first principal component.

    Only days on which every station in the matrix has a value are used; returns ``(share, n_days)``.
    """
    stack = np.asarray(stack, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(np.where(stack > 0, stack, np.nan))
    full = np.isfinite(logs).all(axis=0)
    if full.sum() < min_days:
        raise ValueError(f"only {int(full.sum())} days with every station")
    m = logs[:, full]
    m = (m - m.mean(axis=1, keepdims=True)) / m.std(axis=1, keepdims=True)
    s = np.linalg.svd(m, compute_uv=False)
    return float(s[0] ** 2 / (s ** 2).sum()), int(full.sum())


def fisher_exact_two_sided(a, b, c, d):
    """Two-sided Fisher exact p-value of the 2×2 table [[a, b], [c, d]] (sum of tables at most as probable as the observed one)."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2

    def logp(x):
        return (math.lgamma(n1 + 1) - math.lgamma(x + 1) - math.lgamma(n1 - x + 1) + math.lgamma(n2 + 1) - math.lgamma(k - x + 1) - math.lgamma(n2 - k + x + 1)
                - (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)))
    lo, hi = max(0, k - n2), min(k, n1)
    obs = logp(a)
    total = sum(math.exp(logp(x)) for x in range(lo, hi + 1) if logp(x) <= obs + 1e-9)
    return float(min(1.0, total))


def overlap_table(episode_days, signal_days):
    """2×2 table of days (episode?, signal?) and its Fisher p-value and odds ratio; both arguments are boolean arrays over the same days."""
    e, s = np.asarray(episode_days, bool), np.asarray(signal_days, bool)
    a, b, c, d = int((e & s).sum()), int((e & ~s).sum()), int((~e & s).sum()), int((~e & ~s).sum())
    odds = (a * d) / (b * c) if b * c > 0 else float("inf")
    return {"both": a, "episode_only": b, "signal_only": c, "neither": d, "odds_ratio": float(odds), "p": fisher_exact_two_sided(a, b, c, d),
            "hit_rate": a / (a + b) if a + b else float("nan"), "precision": a / (a + c) if a + c else float("nan")}


def rayleigh(directions_deg):
    """Rayleigh test of circular uniformity: ``(mean direction in degrees, mean resultant length R, p-value)`` (Zar's finite-sample form)."""
    d = np.radians(np.asarray(directions_deg, float))
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        raise ValueError("need at least 5 directions")
    c, s = np.cos(d).sum(), np.sin(d).sum()
    r = math.hypot(c, s) / n
    z = n * r * r
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - (n * r) ** 2)) - (1 + 2 * n))
    return float((math.degrees(math.atan2(s, c)) + 360.0) % 360.0), float(r), float(min(1.0, p)), float(z)

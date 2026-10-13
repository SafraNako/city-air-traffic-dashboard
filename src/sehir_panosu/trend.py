"""Trend estimation on seasonal data: the seasonal Mann-Kendall test and the Theil-Sen slope (Hirsch, Slack & Smith 1982).

Monthly means of an air-quality series have a strong annual cycle, so every calendar month is compared only with the same
calendar month of other years; the Kendall statistics of the twelve seasons add up. The slope is the median of all
within-season pairwise slopes, and its interval comes from resampling whole years (which keeps the correlation between
the months of one year).
"""
from __future__ import annotations

import numpy as np

from .special import norm_sf


def monthly_means(dates, daily, min_days=20):
    """``(month_starts, means)`` of a daily series; a month with fewer than ``min_days`` valid days is NaN."""
    months = dates.astype("datetime64[M]")
    uniq = np.unique(months)
    daily = np.asarray(daily, float)
    out = np.full(len(uniq), np.nan)
    for i, m in enumerate(uniq):
        v = daily[months == m]
        v = v[np.isfinite(v)]
        if len(v) >= min_days:
            out[i] = v.mean()
    return uniq, out


def _season_table(months, values):
    """Arrange monthly values in a ``(years, 12)`` table (NaN where missing)."""
    years = months.astype("datetime64[Y]").astype(int) + 1970
    mon = months.astype(int) % 12
    y0 = years.min()
    table = np.full((years.max() - y0 + 1, 12), np.nan)
    table[years - y0, mon] = values
    return table, y0


def _kendall_s_var(col):
    v = col[np.isfinite(col)]
    n = len(v)
    if n < 2:
        return 0.0, 0.0
    diff = v[None, :] - v[:, None]
    s = float(np.sign(diff[np.triu_indices(n, 1)]).sum())
    _, counts = np.unique(v, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    return s, float(var)


def seasonal_kendall(table):
    """Seasonal Mann-Kendall statistic: ``(S, variance, z, two-sided p)`` over the columns of a ``(years, seasons)`` table."""
    s_tot, v_tot = 0.0, 0.0
    for k in range(table.shape[1]):
        s, v = _kendall_s_var(table[:, k])
        s_tot += s
        v_tot += v
    if v_tot <= 0:
        return s_tot, v_tot, 0.0, 1.0
    z = (s_tot - np.sign(s_tot)) / np.sqrt(v_tot)
    return s_tot, v_tot, float(z), float(2 * norm_sf(abs(z)))


def seasonal_sen_slope(table):
    """Median of the within-season pairwise slopes (units of the values per year)."""
    slopes = []
    years = np.arange(table.shape[0], dtype=float)
    for k in range(table.shape[1]):
        col = table[:, k]
        ok = np.where(np.isfinite(col))[0]
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                slopes.append((col[ok[b]] - col[ok[a]]) / (years[ok[b]] - years[ok[a]]))
    return float(np.median(slopes)) if slopes else float("nan")


def trend_test(months, values, level=0.95, reps=300, seed=0, min_years=4):
    """Seasonal Kendall test, Sen slope and a year-resampling interval for the slope.

    Returns a dict with ``slope`` (per year), ``low``/``high``, ``p``, ``z``, ``years`` (number of years spanned), ``relative``
    (slope as a fraction of the mean per year).
    """
    table, y0 = _season_table(months, values)
    if table.shape[0] < min_years or np.isfinite(table).sum() < 24:
        raise ValueError("too short for a seasonal trend test")
    s, v, z, p = seasonal_kendall(table)
    slope = seasonal_sen_slope(table)
    rng = np.random.default_rng(seed)
    boots = np.empty(reps)
    for b in range(reps):
        idx = rng.integers(0, table.shape[0], table.shape[0])
        # a resampled year keeps its place in time through the regression on its own year label: use the label of the drawn year
        boots[b] = _slope_with_labels(table[idx], idx)
    a = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [a, 1 - a])
    mean = float(np.nanmean(table))
    return {"slope": slope, "low": float(lo), "high": float(hi), "p": p, "z": z, "S": s, "years": int(table.shape[0]), "mean": mean,
            "relative": slope / mean if mean else float("nan"), "start_year": int(y0)}


def _slope_with_labels(table, labels):
    """Sen slope of a table whose rows come from the years ``labels`` (rows drawn twice are skipped: they have no time difference)."""
    slopes = []
    labels = np.asarray(labels, float)
    for k in range(table.shape[1]):
        col = table[:, k]
        ok = np.where(np.isfinite(col))[0]
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                dt = labels[ok[b]] - labels[ok[a]]
                if dt != 0:
                    slopes.append((col[ok[b]] - col[ok[a]]) / dt)
    return float(np.median(slopes)) if slopes else float("nan")


def annual_stats(dates, daily, completeness=0.75):
    """Per calendar year: valid days, completeness, mean, and the 95th percentile of the daily values (only years above the completeness)."""
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    daily = np.asarray(daily, float)
    out = {}
    for y in np.unique(years):
        sel = years == y
        v = daily[sel]
        valid = np.isfinite(v)
        days_in_year = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        out[int(y)] = {"valid_days": int(valid.sum()), "completeness": float(valid.sum() / min(days_in_year, sel.sum() if y == years.max() else days_in_year)),
                       "mean": float(v[valid].mean()) if valid.any() else float("nan"),
                       "p95": float(np.quantile(v[valid], 0.95)) if valid.any() else float("nan"), "complete": bool(valid.sum() >= completeness * days_in_year)}
    return out


def exceedance_days(daily, limit):
    daily = np.asarray(daily, float)
    return int(np.nansum(daily[np.isfinite(daily)] > limit))

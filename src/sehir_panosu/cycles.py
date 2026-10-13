"""Diurnal, weekly and holiday patterns with honest uncertainty.

Intervals come from resampling whole days (so the strong hour-to-hour correlation inside a day is kept) or from paired
differences within weeks (so the seasonal level cancels); the tests simulate the coverage and the false-alarm rate.
"""
from __future__ import annotations

import numpy as np

from .special import t_ppf, t_sf
from .timeseries import Hourly, first_midnight


def day_matrix(h: Hourly, name):
    """``(dates, M)`` with ``M[d, hour]`` the value of day ``d`` at that hour (NaN where missing); the first partial day is skipped."""
    off = first_midnight(h)
    x = h[name][off:]
    nd = len(x) // 24
    dates = (h.start + off * np.timedelta64(1, "h")).astype("datetime64[D]") + np.arange(nd)
    return dates, x[: nd * 24].reshape(nd, 24)


def diurnal_profile(matrix, day_mask=None, reps=200, level=0.95, seed=0, log=False):
    """Mean by hour of day over the selected days with a day-resampling bootstrap interval; returns ``(mean, low, high)`` arrays of 24.

    With ``log=True`` the values are averaged on the log scale (a geometric mean), which suits concentrations.
    """
    m = np.asarray(matrix, float)
    if day_mask is not None:
        m = m[np.asarray(day_mask)]
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.log(np.where(m > 0, m, np.nan))
    if len(m) < 5:
        raise ValueError("need at least 5 days")
    rng = np.random.default_rng(seed)
    with np.errstate(all="ignore"):
        center = np.nanmean(m, axis=0)
        boots = np.empty((reps, 24))
        for b in range(reps):
            boots[b] = np.nanmean(m[rng.integers(0, len(m), len(m))], axis=0)
    a = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [a, 1 - a], axis=0)
    if log:
        return np.exp(center), np.exp(lo), np.exp(hi)
    return center, lo, hi


def weekday_means(dates, daily):
    """Mean of a daily series for each weekday (Monday = 0)."""
    wd = (dates.astype(int) + 3) % 7
    daily = np.asarray(daily, float)
    return np.array([np.nanmean(daily[wd == k]) if np.isfinite(daily[wd == k]).any() else np.nan for k in range(7)])


def week_index(dates):
    """Week number counted from the Monday on or before the first date, one integer per date."""
    d = dates.astype(int)
    monday0 = d[0] - (d[0] + 3) % 7
    return (d - monday0) // 7


def weekend_effect(dates, daily, level=0.95, min_weekdays=4, min_weeks=20):
    """Log-ratio of weekend to weekday days, estimated from paired weeks.

    For every week with at least ``min_weekdays`` valid weekdays and one valid weekend day the log of the mean weekend value minus the
    log of the mean weekday value is taken; the effect is the mean of those differences with a Student-t interval. Returns a dict with
    ``effect`` (log ratio), ``ratio``, ``low``/``high`` (ratio), ``p`` (two-sided, H0: no weekend effect), ``weeks``.
    """
    daily = np.asarray(daily, float)
    wk = week_index(dates)
    wd = (dates.astype(int) + 3) % 7
    diffs = []
    for w in np.unique(wk):
        sel = wk == w
        a = daily[sel & (wd < 5)]
        b = daily[sel & (wd >= 5)]
        a, b = a[np.isfinite(a) & (a > 0)], b[np.isfinite(b) & (b > 0)]
        if len(a) >= min_weekdays and len(b) >= 1:
            diffs.append(np.log(b.mean()) - np.log(a.mean()))
    d = np.array(diffs)
    if len(d) < min_weeks:
        raise ValueError(f"only {len(d)} usable weeks")
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    tcrit = t_ppf(1 - (1 - level) / 2, len(d) - 1)
    t = mean / se
    return {"effect": float(mean), "ratio": float(np.exp(mean)), "low": float(np.exp(mean - tcrit * se)), "high": float(np.exp(mean + tcrit * se)),
            "p": float(2 * t_sf(abs(t), len(d) - 1)), "weeks": int(len(d))}


def holiday_effect(dates, daily, holidays, level=0.95, reference_weeks=(-2, -1, 1, 2), min_events=5):
    """Log-ratio of public holidays that fall on weekdays to the same weekday in the weeks around (non-holiday references only).

    Returns ``dict(effect, ratio, low, high, p, events)`` and the list of ``(date, ratio)`` for the individual holidays.
    """
    daily = np.asarray(daily, float)
    pos = {int(d): i for i, d in enumerate(dates.astype(int))}
    hol = {int(np.datetime64(h, "D").astype(int)) for h in holidays}
    logs, items = [], []
    for d in sorted(hol):
        if d not in pos or (d + 3) % 7 >= 5:
            continue
        v = daily[pos[d]]
        refs = []
        for k in reference_weeks:
            dd = d + 7 * k
            if dd in pos and dd not in hol and np.isfinite(daily[pos[dd]]) and daily[pos[dd]] > 0:
                refs.append(daily[pos[dd]])
        if np.isfinite(v) and v > 0 and len(refs) >= 2:
            r = v / np.mean(refs)
            logs.append(np.log(r))
            items.append((np.datetime64(d, "D"), float(r)))
    l = np.array(logs)
    if len(l) < min_events:
        raise ValueError(f"only {len(l)} usable holidays")
    mean, se = l.mean(), l.std(ddof=1) / np.sqrt(len(l))
    tcrit = t_ppf(1 - (1 - level) / 2, len(l) - 1)
    return ({"effect": float(mean), "ratio": float(np.exp(mean)), "low": float(np.exp(mean - tcrit * se)), "high": float(np.exp(mean + tcrit * se)),
             "p": float(2 * t_sf(abs(mean / se), len(l) - 1)), "events": int(len(l))}, items)


def seasonal_profile(dates, daily):
    """Mean by calendar month of a daily series (index 0 = January)."""
    month = dates.astype("datetime64[M]").astype(int) % 12
    daily = np.asarray(daily, float)
    return np.array([np.nanmean(daily[month == k]) if np.isfinite(daily[month == k]).any() else np.nan for k in range(12)])

"""Data-quality rules for hourly air-quality series: completeness, repeated (stuck) values, spikes, impossible values.

Every rule is checked against planted faults in the tests and in ``sehir dogrulama``. The thresholds are not magic numbers:
the stuck-run length comes from the series' own repeat probability, the spike rule from a robust (median / MAD) scale.
"""
from __future__ import annotations

import warnings

import numpy as np

# values above these (µg/m³, hourly) are physically implausible for urban ambient air
PHYSICAL_MAX = {"PM10": 2000.0, "SO2": 3000.0, "O3": 1000.0, "NO2": 1500.0, "CO": 60000.0}


def completeness(x):
    """Fraction of hours with a value."""
    x = np.asarray(x, float)
    return float(np.isfinite(x).mean()) if len(x) else float("nan")


def completeness_by_year(years, x):
    """``{year: fraction of valid hours}`` for an hourly array and its per-hour calendar year."""
    years = np.asarray(years)
    x = np.asarray(x, float)
    return {int(y): float(np.isfinite(x[years == y]).mean()) for y in np.unique(years)}


def repeat_probability(x):
    """Probability that a valid hour equals the previous valid hour, estimated on the series without its long runs."""
    x = np.asarray(x, float)
    both = np.isfinite(x[1:]) & np.isfinite(x[:-1])
    if both.sum() < 100:
        return float("nan")
    same = (x[1:] == x[:-1]) & both
    return float(same.sum() / both.sum())


def stuck_threshold(x, expected_false=0.5, floor=4):
    """Smallest run length of identical values that would occur by chance fewer than ``expected_false`` times in the series.

    A run of ``k`` equal values needs ``k - 1`` repeats; with repeat probability ``q`` the expected number of such starts in ``n`` valid
    hours is about ``n q^(k-1)``. The probability ``q`` is taken after discarding runs of five or more, so that the faults being
    looked for do not inflate their own threshold.
    """
    x = np.asarray(x, float)
    n = int(np.isfinite(x).sum())
    trimmed = np.where(stuck_mask(x, 5), np.nan, x)
    q = repeat_probability(trimmed)
    if not np.isfinite(q) or q <= 0:
        return floor
    k = 2
    while n * q ** (k - 1) > expected_false and k < 200:
        k += 1
    return max(k, floor)


def stuck_mask(x, min_run):
    """Hours that belong to a run of at least ``min_run`` identical consecutive valid values (gaps break a run)."""
    x = np.asarray(x, float)
    n = len(x)
    out = np.zeros(n, bool)
    if n == 0:
        return out
    ok = np.isfinite(x)
    start = 0
    for i in range(1, n + 1):
        if i == n or not ok[i] or not ok[i - 1] or x[i] != x[i - 1]:
            if ok[start] and i - start >= min_run:
                out[start:i] = True
            start = i
    return out


def spike_mask(x, window=25, k=12.0, floor=None):
    """Isolated high spikes: a value more than ``k`` robust standard deviations above the median of its centred window, while the
    neighbours on both sides are not (so a genuine episode, where neighbours are high too, is not flagged).

    The robust scale is the median absolute deviation of the window times 1.4826, with a floor of ``floor`` (default 5 % of the window
    median) so that very flat windows do not flag every wiggle.
    """
    x = np.asarray(x, float)
    n = len(x)
    out = np.zeros(n, bool)
    half = window // 2
    pad = np.concatenate([np.full(half, np.nan), x, np.full(half, np.nan)])
    from numpy.lib.stride_tricks import sliding_window_view
    view = sliding_window_view(pad, window)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # windows without any valid value
        med = np.nanmedian(view, axis=1)
        mad = np.nanmedian(np.abs(view - med[:, None]), axis=1) * 1.4826
    scale = np.maximum(mad, (0.05 if floor is None else floor) * np.abs(med))
    scale = np.where(scale > 0, scale, np.nan)
    z = (x - med) / scale
    cand = np.isfinite(z) & (z > k)
    prev = np.concatenate([[np.nan], x[:-1]])
    nxt = np.concatenate([x[1:], [np.nan]])
    with np.errstate(invalid="ignore"):
        isolated = ~((prev - med) / scale > k / 2) & ~((nxt - med) / scale > k / 2)
    out[cand & isolated] = True
    return out


def physical_mask(pollutant, x):
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore"):
        return (x < 0) | (x > PHYSICAL_MAX[pollutant])


def flag_series(pollutant, x):
    """Boolean masks ``{stuck, spike, impossible}`` for one hourly series and the stuck threshold used."""
    x = np.asarray(x, float)
    run = stuck_threshold(x)
    return {"stuck": stuck_mask(x, run), "spike": spike_mask(x), "impossible": physical_mask(pollutant, x)}, run


def clean(pollutant, x):
    """The series with every flagged hour set to NaN, and the flags."""
    flags, run = flag_series(pollutant, x)
    bad = flags["stuck"] | flags["spike"] | flags["impossible"]
    out = np.where(bad, np.nan, np.asarray(x, float))
    return out, flags, run

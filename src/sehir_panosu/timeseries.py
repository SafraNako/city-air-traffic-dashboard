"""Hourly time-series container and the calendar helpers the analyses share.

All times are naive local clock times of Istanbul. Turkey has been on a constant UTC+3 since September 2016, so on the
2019-2026 span used here local time is a fixed shift of UTC and has no daylight-saving jumps.
"""
from __future__ import annotations

import warnings

import numpy as np

HOUR = np.timedelta64(1, "h")


def to_hour(value):
    """A datetime64[h] from a string, date or datetime64 (strings may carry minutes or seconds, which are dropped)."""
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[h]")
    return np.datetime64(str(value)[:13] if "T" in str(value) or " " in str(value) else str(value) + "T00", "h")


def to_day(value):
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[D]")
    return np.datetime64(str(value)[:10], "D")


class Hourly:
    """Equally spaced hourly series sharing one time axis; missing hours are NaN."""

    def __init__(self, start, n, arrays=None):
        self.start = to_hour(start)
        self.n = int(n)
        self.arrays = {}
        for k, v in (arrays or {}).items():
            self.set(k, v)

    def set(self, name, values):
        v = np.asarray(values, dtype=float)
        if v.shape != (self.n,):
            raise ValueError(f"series {name!r} has length {v.shape}, expected {self.n}")
        self.arrays[name] = v

    def __getitem__(self, name):
        return self.arrays[name]

    def __contains__(self, name):
        return name in self.arrays

    def __len__(self):
        return self.n

    @property
    def end(self):
        """The hour after the last one."""
        return self.start + self.n * HOUR

    def times(self):
        return self.start + np.arange(self.n) * HOUR

    def dates(self):
        return self.times().astype("datetime64[D]")

    def hour_of_day(self):
        return (self.times() - self.dates().astype("datetime64[h]")).astype(int)

    def weekday(self):
        """Monday = 0 … Sunday = 6."""
        return (self.dates().astype(int) + 3) % 7

    def month(self):
        return self.dates().astype("datetime64[M]").astype(int) % 12 + 1

    def year(self):
        return self.dates().astype("datetime64[Y]").astype(int) + 1970

    def index_of(self, when):
        i = int((to_hour(when) - self.start) / HOUR)
        return i

    def slice(self, start=None, end=None):
        """The hours with start <= t < end (clipped to the series)."""
        i0 = 0 if start is None else max(0, self.index_of(start))
        i1 = self.n if end is None else min(self.n, self.index_of(end))
        i1 = max(i0, i1)
        return Hourly(self.start + i0 * HOUR, i1 - i0, {k: v[i0:i1] for k, v in self.arrays.items()})

    def copy(self):
        return Hourly(self.start, self.n, {k: v.copy() for k, v in self.arrays.items()})

    @classmethod
    def from_records(cls, times, columns, start=None, end=None):
        """Place records (times as datetime64[h], later duplicates win) on a regular grid covering them, or ``[start, end)``."""
        times = np.asarray(times, dtype="datetime64[h]")
        lo = times.min() if start is None else to_hour(start)
        hi = (times.max() + HOUR) if end is None else to_hour(end)
        n = int((hi - lo) / HOUR)
        idx = ((times - lo) / HOUR).astype(int)
        keep = (idx >= 0) & (idx < n)
        out = cls(lo, n)
        for name, values in columns.items():
            arr = np.full(n, np.nan)
            arr[idx[keep]] = np.asarray(values, float)[keep]
            out.set(name, arr)
        return out

    def align_to(self, other):
        """A copy of this series laid on ``other``'s time axis (NaN outside the overlap)."""
        out = Hourly(other.start, other.n)
        shift = int((self.start - other.start) / HOUR)
        for k, v in self.arrays.items():
            arr = np.full(other.n, np.nan)
            lo, hi = max(0, shift), min(other.n, shift + self.n)
            if hi > lo:
                arr[lo:hi] = v[lo - shift:hi - shift]
            out.set(k, arr)
        return out


def rolling_mean(x, window, min_count):
    """Trailing mean over ``window`` hours that needs at least ``min_count`` valid values (NaN otherwise)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(ok)])
    n = len(x)
    lo = np.maximum(np.arange(1, n + 1) - window, 0)
    hi = np.arange(1, n + 1)
    cnt = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (csum[hi] - csum[lo]) / cnt
    out[cnt < min_count] = np.nan
    return out


def daily_aggregate(x, min_count=18, how="mean"):
    """Mean (or max) of each calendar day of an hourly array starting at midnight; days with too few valid hours are NaN."""
    x = np.asarray(x, float)
    days = len(x) // 24
    block = x[: days * 24].reshape(days, 24)
    cnt = np.isfinite(block).sum(axis=1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # days without any valid hour
        out = np.nanmean(block, axis=1) if how == "mean" else np.nanmax(block, axis=1)
    out[cnt < min_count] = np.nan
    return out


def first_midnight(h: Hourly):
    """Index of the first midnight at or after the start, so that daily blocks line up with calendar days."""
    hod = int((h.start - h.start.astype("datetime64[D]").astype("datetime64[h]")) / HOUR)
    return (24 - hod) % 24


def daily(h: Hourly, name, min_count=18, how="mean"):
    """``(dates, values)`` of a daily mean or maximum of one series; the first partial day is skipped."""
    off = first_midnight(h)
    values = daily_aggregate(h[name][off:], min_count, how)
    dates = (h.start + off * HOUR).astype("datetime64[D]") + np.arange(len(values))
    return dates, values


def merge_hourly(base: Hourly, new: Hourly):
    """The union of two series on one grid; finite values of ``new`` override those of ``base`` (the live service wins for the recent past)."""
    lo = min(base.start, new.start)
    hi = max(base.end, new.end)
    out = Hourly(lo, int((hi - lo) / HOUR))
    for name in dict.fromkeys(list(base.arrays) + list(new.arrays)):
        arr = np.full(out.n, np.nan)
        for part in (base, new):
            if name in part:
                a = int((part.start - lo) / HOUR)
                seg = part[name]
                arr[a:a + part.n] = np.where(np.isfinite(seg), seg, arr[a:a + part.n])
        out.set(name, arr)
    return out

"""Traffic-index analyses: hourly profiles, holiday and rain effects, and detection of days that are quieter than the calendar explains.

The traffic index is the municipality's congestion index (0 free-flowing … 100 gridlock), not a vehicle count.
"""
from __future__ import annotations

import numpy as np

from . import cycles, normalize
from .timeseries import Hourly


def hourly_profiles(h: Hourly, name, holidays=()):
    """Mean daily curves for weekdays, Saturdays, Sundays and weekday public holidays; each is ``(mean, low, high)`` arrays of 24."""
    dates, m = cycles.day_matrix(h, name)
    wd = (dates.astype(int) + 3) % 7
    hol = {int(np.datetime64(x, "D").astype(int)) for x in holidays}
    is_hol = np.array([int(d) in hol for d in dates.astype(int)])
    masks = {"weekday": (wd < 5) & ~is_hol, "saturday": (wd == 5) & ~is_hol, "sunday": (wd == 6) & ~is_hol, "holiday": (wd < 5) & is_hol}
    out = {}
    for k, mask in masks.items():
        if mask.sum() >= 5:
            out[k] = cycles.diurnal_profile(m, mask, reps=150)
    return out


def _robust_fit(X, y, rounds=3, clip=3.5):
    """OLS that drops points beyond ``clip`` robust standard deviations and refits (so a few holidays do not bend the baseline)."""
    use = np.isfinite(y) & np.isfinite(X).all(axis=1)
    for _ in range(rounds):
        f = normalize.ols(X[use], y[use])
        resid = np.full(len(y), np.nan)
        resid[np.isfinite(y) & np.isfinite(X).all(axis=1)] = (y - X @ f["beta"])[np.isfinite(y) & np.isfinite(X).all(axis=1)]
        sd = 1.4826 * np.nanmedian(np.abs(resid[use] - np.nanmedian(resid[use])))
        keep = use & (np.abs(resid) <= clip * sd)
        if keep.sum() == use.sum():
            break
        use = keep
    return f, resid, sd, use


def quiet_days(dates, daily_index, holidays=(), threshold=3.5, min_run=1):
    """Days whose traffic is far below what weekday, month and public holidays explain.

    A log-linear calendar model is fitted robustly; a day is *quiet* if its residual is more than ``threshold`` robust standard
    deviations below zero and it is not itself a listed public holiday. Returns ``dict(events=[(first date, last date, mean ratio)], z, ratio)``
    where ``ratio`` is observed over expected traffic per day.
    """
    y = np.log(np.where(np.asarray(daily_index, float) > 0, daily_index, np.nan))
    X, names = normalize.build_design(dates, holidays)
    f, resid, sd, use = _robust_fit(X, y)
    z = resid / sd
    hol = {int(np.datetime64(h, "D").astype(int)) for h in holidays}
    flag = np.isfinite(z) & (z < -threshold) & np.array([int(d) not in hol for d in dates.astype(int)])
    events, i = [], 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j + 1 < len(flag) and flag[j + 1]:
                j += 1
            if j - i + 1 >= min_run:
                events.append((dates[i], dates[j], float(np.exp(np.nanmean(resid[i:j + 1])))))
            i = j + 1
        else:
            i += 1
    return {"events": events, "z": z, "ratio": np.exp(resid), "sd": float(sd)}


def rain_effect(dates, daily_index, rain_mm, holidays=(), threshold_mm=1.0, lags=7):
    """Log-change of the daily traffic index on rainy days (>= ``threshold_mm``), net of weekday, month and holidays; Newey-West interval."""
    y = np.log(np.where(np.asarray(daily_index, float) > 0, daily_index, np.nan))
    rain = (np.asarray(rain_mm, float) >= threshold_mm).astype(float)
    rain[~np.isfinite(rain_mm)] = np.nan
    X, names = normalize.build_design(dates, holidays, extra={"rain": rain})
    f, use = normalize.fit_complete_rows(X, y, lags=lags)
    i = names.index("rain")
    b, se = f["beta"][i], f["se"][i]
    return {"effect": float(b), "ratio": float(np.exp(b)), "low": float(np.exp(b - 1.96 * se)), "high": float(np.exp(b + 1.96 * se)), "p": float(f["p"][i]),
            "rainy_days": int(np.nansum(rain[use])), "days": int(use.sum())}


def typical_curve(h: Hourly, name, holidays=()):
    """Median and quartiles of the traffic index for every weekday and hour (holidays excluded): arrays of shape (7, 24)."""
    dates, m = cycles.day_matrix(h, name)
    wd = (dates.astype(int) + 3) % 7
    hol = {int(np.datetime64(x, "D").astype(int)) for x in holidays}
    keep = np.array([int(d) not in hol for d in dates.astype(int)])
    out = {"median": np.full((7, 24), np.nan), "q25": np.full((7, 24), np.nan), "q75": np.full((7, 24), np.nan)}
    for k in range(7):
        rows = m[(wd == k) & keep]
        if len(rows) >= 5:
            out["median"][k] = np.nanmedian(rows, axis=0)
            out["q25"][k], out["q75"][k] = np.nanquantile(rows, [0.25, 0.75], axis=0)
    return out


def now_panel(h: Hourly, name, holidays=(), hours=72):
    """The latest ``hours`` of the index with the typical median and quartile band for the same weekday and hour, and the latest value's percentile."""
    typ = typical_curve(h, name, holidays)
    last = h.n
    while last > 0 and not np.isfinite(h[name][last - 1]):
        last -= 1
    first = max(0, last - hours)
    times = h.times()[first:last]
    wd = ((times.astype("datetime64[D]").astype(int) + 3) % 7)
    hr = (times - times.astype("datetime64[D]").astype("datetime64[h]")).astype(int)
    dates, m = cycles.day_matrix(h, name)
    wdays = (dates.astype(int) + 3) % 7
    latest = float(h[name][last - 1])
    earlier = dates != times[-1].astype("datetime64[D]")                       # the latest day itself is not part of its own reference
    same = m[(wdays == wd[-1]) & earlier, hr[-1]]
    same = same[np.isfinite(same)]
    pct = float((same < latest).mean() + 0.5 * (same == latest).mean()) if len(same) else float("nan")
    return {"times": times, "values": h[name][first:last], "median": typ["median"][wd, hr], "q25": typ["q25"][wd, hr], "q75": typ["q75"][wd, hr], "latest": latest,
            "latest_time": str(times[-1]), "percentile": pct}

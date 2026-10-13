"""Regression machinery for weather normalisation and for the traffic-pollution link.

Ordinary least squares with a Newey-West (Bartlett kernel) covariance, because daily pollution residuals are autocorrelated;
a design-matrix builder for calendar effects (weekday, month, holidays) and weather; and a *placebo-window* test: the average
residual of a period of interest is ranked among the average residuals of all other windows of the same length, which needs
no distributional assumption.
"""
from __future__ import annotations

import numpy as np

from .special import norm_sf


def ols(X, y, lags=0):
    """OLS fit; returns ``dict(beta, se, t, p, resid, r2, n, fitted)`` using Newey-West errors with ``lags`` lags (0 = plain)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    if lags <= 0:
        s2 = resid @ resid / max(1, n - k)
        cov = s2 * xtx_inv
    else:
        u = X * resid[:, None]
        meat = u.T @ u
        for l in range(1, lags + 1):
            w = 1.0 - l / (lags + 1.0)
            gamma = u[l:].T @ u[:-l]
            meat += w * (gamma + gamma.T)
        cov = xtx_inv @ meat @ xtx_inv * n / max(1, n - k)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = np.array([2.0 * norm_sf(abs(v)) if np.isfinite(v) else np.nan for v in t])
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"beta": beta, "se": se, "t": t, "p": p, "resid": resid, "fitted": fitted, "r2": float(1 - resid @ resid / ss_tot) if ss_tot > 0 else float("nan"), "n": int(n)}


def calendar_columns(dates, holidays=()):
    """Columns (and their names) for weekday dummies (Monday base), month dummies (January base) and a weekday-holiday indicator."""
    d = dates.astype(int)
    wd = (d + 3) % 7
    month = dates.astype("datetime64[M]").astype(int) % 12
    hol = {int(np.datetime64(h, "D").astype(int)) for h in holidays}
    cols, names = [], []
    for k in range(1, 7):
        cols.append((wd == k).astype(float))
        names.append(f"weekday{k}")
    for k in range(1, 12):
        cols.append((month == k).astype(float))
        names.append(f"month{k + 1}")
    cols.append(np.array([1.0 if int(x) in hol and (int(x) + 3) % 7 < 5 else 0.0 for x in d]))
    names.append("holiday")
    return cols, names


def build_design(dates, holidays=(), extra=None, intercept=True):
    """Design matrix of calendar effects plus the named extra columns; returns ``(X, names)``."""
    cols, names = calendar_columns(dates, holidays)
    if intercept:
        cols = [np.ones(len(dates))] + cols
        names = ["const"] + names
    for k, v in (extra or {}).items():
        cols.append(np.asarray(v, float))
        names.append(k)
    return np.column_stack(cols), names


def fit_complete_rows(X, y, train=None, lags=0, min_rows=100):
    """OLS on the rows without NaN (and within ``train`` if given); returns ``(fit, usable_row_mask)``."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    use = ok if train is None else ok & train
    if use.sum() < min_rows:
        raise ValueError(f"only {int(use.sum())} complete rows")
    return ols(X[use], y[use], lags), use


def window_effect(resid, dates, start, end, min_valid_fraction=0.7):
    """Mean residual in ``[start, end]`` and its rank among all windows of the same length (placebo windows).

    ``resid`` is a daily residual series (NaN where unknown). Returns ``dict(mean, placebo_mean, placebo_sd, z, rank, windows, p, valid_days)``
    where ``p`` is the share of placebo windows at least as extreme (two-sided) and ``z`` the effect in placebo standard deviations.
    """
    resid = np.asarray(resid, float)
    s = int(np.argmax(dates >= np.datetime64(start, "D")))
    e = int(np.argmax(dates > np.datetime64(end, "D"))) if (dates > np.datetime64(end, "D")).any() else len(dates)
    length = e - s
    if length < 3:
        raise ValueError("window too short")
    ok = np.isfinite(resid)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, resid, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(ok)])
    starts = np.arange(0, len(resid) - length + 1)
    cnt = ccnt[starts + length] - ccnt[starts]
    means = np.where(cnt >= min_valid_fraction * length, (csum[starts + length] - csum[starts]) / np.maximum(cnt, 1), np.nan)
    mine = means[s] if s < len(means) else float("nan")
    # placebo windows must not overlap the window of interest
    far = (starts + length <= s) | (starts >= e)
    pl = means[far & np.isfinite(means)]
    if not np.isfinite(mine) or len(pl) < 20:
        raise ValueError("not enough valid data in or around the window")
    return {"mean": float(mine), "placebo_mean": float(pl.mean()), "placebo_sd": float(pl.std(ddof=1)), "z": float((mine - pl.mean()) / pl.std(ddof=1)),
            "rank": int((pl < mine).sum() + 1), "windows": int(len(pl)), "p": float(min(1.0, 2 * min((pl <= mine).mean(), (pl >= mine).mean()) + 1.0 / len(pl))),
            "valid_days": int(cnt[s])}


def lagged(x, k):
    """``x`` shifted so that element ``i`` holds ``x[i - k]`` (NaN-filled at the start)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if k >= 0:
        out[k:] = x[: len(x) - k] if k else x
    else:
        out[:k] = x[-k:]
    return out

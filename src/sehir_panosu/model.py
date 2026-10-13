"""Comparing a gridded model (CAMS) with station measurements: bias, error, correlation with an interval that respects autocorrelation."""
from __future__ import annotations

import math

import numpy as np

from .special import norm_ppf


def lag1_autocorr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 0.0
    a, b = x[:-1] - x.mean(), x[1:] - x.mean()
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def effective_n(x, y):
    """Effective sample size of paired series with autocorrelated errors: ``n (1 - rho) / (1 + rho)`` with ``rho`` the lag-1 autocorrelation of the product of the centred series."""
    ok = np.isfinite(x) & np.isfinite(y)
    xc, yc = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    rho = max(0.0, min(0.99, lag1_autocorr(xc * yc)))
    return max(4.0, ok.sum() * (1 - rho) / (1 + rho))


def compare(obs, mod, level=0.95):
    """Metrics of a model against observations (same units): ``n``, mean obs/model, ``bias`` (model - obs), ``nmb`` (normalised mean bias),
    ``rmse``, Pearson ``r`` with a Fisher-z interval based on the effective sample size, and ``slope`` of model on observation."""
    obs, mod = np.asarray(obs, float), np.asarray(mod, float)
    ok = np.isfinite(obs) & np.isfinite(mod)
    n = int(ok.sum())
    if n < 30:
        raise ValueError(f"only {n} paired values")
    o, m = obs[ok], mod[ok]
    r = float(np.corrcoef(o, m)[0, 1])
    neff = effective_n(obs, mod)
    z = np.arctanh(min(0.999999, abs(r))) * np.sign(r)
    half = norm_ppf(1 - (1 - level) / 2) / math.sqrt(max(neff - 3.0, 1.0))
    return {"n": n, "obs_mean": float(o.mean()), "mod_mean": float(m.mean()), "bias": float((m - o).mean()), "nmb": float((m - o).sum() / o.sum()),
            "rmse": float(np.sqrt(np.mean((m - o) ** 2))), "r": r, "r_low": float(np.tanh(z - half)), "r_high": float(np.tanh(z + half)), "n_eff": float(neff),
            "slope": float(np.polyfit(o, m, 1)[0])}

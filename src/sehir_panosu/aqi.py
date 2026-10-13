"""The air-quality index (AQI) the İBB service publishes, reconstructed from its own output and checked against it.

The service returns, next to the concentrations, a sub-index for every pollutant and an overall index. Fitting the
published numbers shows that

* each pollutant is first averaged the way the US EPA scale does it: **PM10 over the trailing 24 h, O3 and CO over the
  trailing 8 h, SO2 and NO2 hourly** (the correlation of the sub-index with the trailing 24 h PM10 mean is 0.985 against
  0.67 with the hourly value; for O3 it is 0.9994 with the 8 h mean against 0.80 hourly);
* the averaged concentration is mapped to the sub-index by a continuous piecewise-linear function with the EPA class
  limits 50 / 100 / 150 / 200 / 300 / 400 / 500 and the EPA breakpoints converted to µg/m³;
* the overall index is the largest sub-index.

The breakpoints are not documented by the service. ``fit_knots`` recovers them from the published values; the table
below holds those fitted values where the data reach them and the EPA values (converted at the gas conversion factor the fit
implies) beyond that. ``agreement`` measures how well the table reproduces the published index.
"""
from __future__ import annotations

import numpy as np

from .timeseries import rolling_mean

POLLUTANTS = ("PM10", "SO2", "O3", "NO2", "CO")
LEVELS = (0, 50, 100, 150, 200, 300, 400, 500)

# (window in hours, minimum valid hours) of the trailing mean applied before the breakpoint map: at least half the window
AGGREGATION = {"PM10": (24, 12), "SO2": (1, 1), "NO2": (1, 1), "CO": (8, 4), "O3": (8, 4)}

# concentration (µg/m³) at the index levels 50, 100, 150, 200, 300, 400, 500
KNOTS = {
    "PM10": (54.0, 154.0, 254.0, 354.0, 424.0, 504.0, 604.0),
    "NO2": (102.0, 192.0, 687.4, 1246.0, 2398.0, 3166.0, 3934.0),
    "SO2": (92.4, 199.7, 490.3, 805.6, 1606.6, 2138.6, 2670.6),
    "CO": (5193.6, 11092.0, 14632.0, 18172.0, 35872.0, 47672.0, 59472.0),
    "O3": (108.0, 247.5, 328.0, 408.0, 808.0, 1008.0, 1208.0),
}
# how many of the knots lie inside the range the data reach (the rest are the converted EPA values, an assumption)
IDENTIFIED = {"PM10": 3, "NO2": 3, "SO2": 3, "CO": 1, "O3": 2}

CATEGORIES = (
    (50, "Good", "İyi", "#13a261"),
    (100, "Moderate", "Orta", "#ffcc16"),
    (150, "Unhealthy for sensitive groups", "Hassas gruplar için sağlıksız", "#EC7C1E"),
    (200, "Unhealthy", "Sağlıksız", "#BC0101"),
    (300, "Very unhealthy", "Çok sağlıksız", "#AE0B54"),
    (500, "Hazardous", "Tehlikeli", "#950F35"),
)


def category(index):
    """``(English name, Turkish name, colour)`` of an AQI value (the colours are the ones the service itself returns)."""
    if not np.isfinite(index):
        return ("No data", "Veri yok", "#9ca3af")
    for upper, en, tr, color in CATEGORIES:
        if index <= upper + 0.5:
            return en, tr, color
    return CATEGORIES[-1][1:]


def aggregate(pollutant, values):
    """The trailing mean the AQI of this pollutant is computed from."""
    window, minimum = AGGREGATION[pollutant]
    return np.asarray(values, float) if window == 1 else rolling_mean(values, window, minimum)


def subindex(pollutant, conc, knots=None):
    """Sub-index of an already averaged concentration (NaN stays NaN; values beyond the last knot are capped at 500)."""
    k = KNOTS[pollutant] if knots is None else knots
    conc = np.asarray(conc, float)
    out = np.interp(conc, (0.0,) + tuple(k), LEVELS[: len(k) + 1], right=float(LEVELS[len(k)]))
    out[~np.isfinite(conc) | (conc < 0)] = np.nan
    return out


def sub_indices(arrays):
    """Sub-index series for every pollutant present in ``arrays`` (a mapping name -> hourly concentration array)."""
    return {p: subindex(p, aggregate(p, arrays[p])) for p in POLLUTANTS if p in arrays}


def overall(subs):
    """``(index, dominant pollutant number)``: the largest sub-index per hour and which pollutant it came from (-1 where none)."""
    names = [p for p in POLLUTANTS if p in subs]
    stack = np.vstack([subs[p] for p in names])
    valid = np.isfinite(stack).any(axis=0)
    filled = np.where(np.isfinite(stack), stack, -np.inf)
    idx = np.where(valid, filled.max(axis=0), np.nan)
    dom = np.where(valid, filled.argmax(axis=0), -1)
    return idx, dom, names


def aqi_series(arrays):
    """The overall AQI and the dominant pollutant (as a name array) from hourly concentrations."""
    subs = sub_indices(arrays)
    idx, dom, names = overall(subs)
    dominant = np.array([names[i] if i >= 0 else "" for i in dom], dtype=object)
    return idx, dominant, subs


def agreement(predicted, published):
    """How well a computed index reproduces the published one: fractions equal after rounding, within ±1 and ±2, and the largest gap."""
    predicted, published = np.asarray(predicted, float), np.asarray(published, float)
    ok = np.isfinite(predicted) & np.isfinite(published)
    if not ok.any():
        return {"n": 0, "exact": float("nan"), "within1": float("nan"), "within2": float("nan"), "max_abs": float("nan"), "mean_diff": float("nan")}
    d = np.rint(predicted[ok]) - published[ok]
    return {"n": int(ok.sum()), "exact": float(np.mean(d == 0)), "within1": float(np.mean(np.abs(d) <= 1)), "within2": float(np.mean(np.abs(d) <= 2)),
            "max_abs": float(np.abs(d).max()), "mean_diff": float(d.mean())}


def fit_knots(conc, published, guess, steps=80):
    """Recover the breakpoints that map averaged concentrations to published sub-indices.

    ``guess`` holds starting values for the knots at the index levels 50, 100, … (as many as are identifiable from the data);
    coordinate descent minimises the squared error of the continuous piecewise-linear map. Returns ``(knots, rmse)``.
    """
    x = np.asarray(conc, float)
    y = np.asarray(published, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    best = np.array(guess, float)
    levels = LEVELS[: len(best) + 1]

    def loss(k):
        return float(np.mean((np.interp(x, (0.0,) + tuple(k), levels, right=float(levels[-1])) - y) ** 2))

    cur = loss(best)
    step = best * 0.1
    for _ in range(steps):
        improved = False
        for i in range(len(best)):
            for sign in (1.0, -1.0):
                trial = best.copy()
                trial[i] += sign * step[i]
                if trial[0] > 0 and np.all(np.diff(trial) > 0):
                    value = loss(trial)
                    if value < cur - 1e-12:
                        best, cur, improved = trial, value, True
        if not improved:
            step = step * 0.5
            if np.all(step < 1e-6):
                break
    return best, float(np.sqrt(cur))

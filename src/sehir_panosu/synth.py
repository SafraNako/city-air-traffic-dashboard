"""Synthetic series with a planted truth, used only to validate the methods (never shown as real data).

``make_series`` builds an hourly concentration-like series ``exp(log base + diurnal + weekly + seasonal + trend + weather effect + AR(1) noise)``;
``plant_faults`` adds stuck runs and spikes and returns where they are; ``traffic_no2`` links a traffic series to a concentration
series with a known elasticity and a weather confounder.
"""
from __future__ import annotations

import numpy as np

from .timeseries import Hourly


def diurnal_shape(kind="no2"):
    """Multiplicative hour-of-day shape (mean 1 on the log scale): two rush-hour peaks ("no2"), an afternoon peak ("o3") or flat."""
    h = np.arange(24)
    if kind == "no2":
        s = 0.30 * np.exp(-0.5 * ((h - 8) / 1.8) ** 2) + 0.35 * np.exp(-0.5 * ((h - 19) / 2.2) ** 2) - 0.25 * np.exp(-0.5 * ((h - 3) / 3.0) ** 2)
    elif kind == "o3":
        s = 0.45 * np.exp(-0.5 * ((h - 15) / 3.5) ** 2) - 0.30 * np.exp(-0.5 * ((h - 5) / 3.0) ** 2)
    else:
        s = np.zeros(24)
    return s - s.mean()


def ar1(n, phi, sigma, rng):
    """Stationary AR(1) noise with marginal standard deviation ``sigma``."""
    e = rng.normal(0.0, sigma * np.sqrt(1 - phi * phi), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def make_series(days, rng, start="2019-01-01T00", base=40.0, weekend=-0.20, seasonal=0.25, trend_per_year=0.0, sigma=0.35, phi=0.9, kind="no2",
                covariate=None, covariate_beta=0.0):
    """An hourly ``Hourly`` with the series ``x``, the planted log-effects as attributes of the dict ``truth``.

    ``weekend`` is the planted log-change of Saturdays and Sundays, ``trend_per_year`` the planted log-trend, ``covariate``
    (an array of the series' length) enters with the log-coefficient ``covariate_beta``.
    """
    n = int(days) * 24
    h = Hourly(start, n)
    wd = h.weekday()
    month = h.month()
    t_years = np.arange(n) / (24 * 365.25)
    log = (np.log(base) + diurnal_shape(kind)[h.hour_of_day()] + weekend * (wd >= 5) + seasonal * np.cos(2 * np.pi * (month - 1) / 12.0) + trend_per_year * t_years
           + ar1(n, phi, sigma, rng))
    if covariate is not None:
        log = log + covariate_beta * np.asarray(covariate, float)
    h.set("x", np.exp(log))
    return h


def plant_faults(x, rng, stuck_runs=6, stuck_length=(8, 24), spikes=20, spike_factor=(8.0, 15.0)):
    """Add stuck runs and isolated spikes to a copy of ``x``; returns ``(faulty, stuck_truth, spike_truth)`` boolean masks."""
    out = np.array(x, float)
    n = len(out)
    stuck = np.zeros(n, bool)
    spike = np.zeros(n, bool)
    used = np.zeros(n + 100, bool)
    for _ in range(stuck_runs):
        for _try in range(50):
            ln = int(rng.integers(stuck_length[0], stuck_length[1] + 1))
            a = int(rng.integers(50, n - ln - 50))
            if not used[a - 30:a + ln + 30].any():
                out[a:a + ln] = round(float(out[a]), 1)
                stuck[a:a + ln] = True
                used[a - 30:a + ln + 30] = True
                break
    for _ in range(spikes):
        for _try in range(50):
            a = int(rng.integers(50, n - 50))
            if not used[a - 30:a + 30].any():
                out[a] = out[a] * float(rng.uniform(*spike_factor))
                spike[a] = True
                used[a - 30:a + 30] = True
                break
    return out, stuck, spike


def traffic_no2(days, rng, elasticity=0.5, wind_beta=-0.4, start="2019-01-01T00", drop_window=None, drop_depth=0.5, sigma=0.15, phi=0.7):
    """Daily traffic index and NO2 with a planted elasticity.

    Returns ``dict(dates, traffic, no2, wind, truth)``: wind speed is a confounder that lowers NO2 (``wind_beta`` on the log scale) and is
    independent of traffic; ``drop_window=(i0, i1)`` multiplies traffic by ``drop_depth`` on those days (a lockdown).
    """
    n = int(days)
    dates = np.datetime64(start[:10]) + np.arange(n)
    wd = (dates.astype(int) + 3) % 7
    weekly = np.where(wd >= 5, 0.7, 1.0)
    traffic = 45.0 * weekly * np.exp(ar1(n, 0.6, 0.12, rng)) * (1 + 0.1 * np.cos(2 * np.pi * np.arange(n) / 365.25))
    if drop_window:
        traffic[drop_window[0]:drop_window[1]] *= drop_depth
    wind = np.exp(rng.normal(np.log(12.0), 0.5, n) * 0.6 + ar1(n, 0.5, 0.2, rng) * 0.4) / 1.0
    no2 = 45.0 * (traffic / 45.0) ** elasticity * (wind / 12.0) ** wind_beta * np.exp(ar1(n, phi, sigma, rng))
    return {"dates": dates, "traffic": traffic, "no2": no2, "wind": wind, "truth": {"elasticity": elasticity, "wind_beta": wind_beta}}

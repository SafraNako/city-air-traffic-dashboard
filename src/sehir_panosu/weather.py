"""Daily weather predictors built from the hourly ERA5 series."""
from __future__ import annotations

import numpy as np

from .timeseries import Hourly, daily


def daily_weather(w: Hourly, min_count=18):
    """Daily means of the weather variables plus the rain total and the mean wind direction; returns ``(dates, dict)``."""
    out = {}
    dates = None
    for name in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure", "boundary_layer_height", "shortwave_radiation"):
        if name in w:
            dates, out[name] = daily(w, name, min_count)
    if "precipitation" in w:
        dates, mean = daily(w, "precipitation", min_count)
        out["precipitation"] = mean * 24.0                          # mean mm per hour × 24 = mm per day
    if "wind_direction_10m" in w and "wind_speed_10m" in w:
        rad = np.radians(w["wind_direction_10m"])
        u = Hourly(w.start, w.n, {"s": np.sin(rad) * w["wind_speed_10m"], "c": np.cos(rad) * w["wind_speed_10m"]})
        _, s = daily(u, "s", min_count)
        _, c = daily(u, "c", min_count)
        out["wind_direction"] = (np.degrees(np.arctan2(s, c)) + 360.0) % 360.0
        out["wind_u"], out["wind_v"] = s, c
    return dates, out


def wind_sector(direction, sectors=8):
    """Integer sector (0 = north, going clockwise) of directions in degrees; NaN stays -1."""
    d = np.asarray(direction, float)
    s = np.floor(((d + 180.0 / sectors) % 360.0) / (360.0 / sectors)).astype(float)
    s[~np.isfinite(d)] = -1
    return s.astype(int)


SECTOR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def met_predictors(dates, wdates, wv):
    """Weather columns for a daily regression on ``dates``: log wind speed (and its lag), log boundary-layer height, temperature, rain flag,
    humidity, log radiation, and the first harmonic of the wind direction; NaN where the weather series has no value."""
    pos = {int(d): i for i, d in enumerate(wdates.astype(int))}
    idx = np.array([pos.get(int(d), -1) for d in dates.astype(int)])
    ok = idx >= 0

    def pick(name):
        out = np.full(len(dates), np.nan)
        out[ok] = np.asarray(wv[name], float)[idx[ok]]
        return out

    wind = pick("wind_speed_10m")
    lag = np.concatenate([[np.nan], wind[:-1]])
    direction = np.radians(pick("wind_direction"))
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"log_wind": np.log(np.maximum(wind, 0.5)), "log_wind_lag1": np.log(np.maximum(lag, 0.5)), "log_blh": np.log(np.maximum(pick("boundary_layer_height"), 50.0)),
                "temp": pick("temperature_2m"), "rain": np.where(np.isfinite(pick("precipitation")), (pick("precipitation") >= 1.0).astype(float), np.nan),
                "humidity": pick("relative_humidity_2m"), "log_rad": np.log(np.maximum(pick("shortwave_radiation"), 1.0)),
                "dir_sin": np.sin(direction), "dir_cos": np.cos(direction)}

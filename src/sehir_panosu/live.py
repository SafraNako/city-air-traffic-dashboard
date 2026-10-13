"""The live part: the last few days of air quality and traffic fetched from the İBB services and merged into the data snapshot."""
from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import replace

import numpy as np

from . import sources
from .dataset import Dataset
from .timeseries import merge_hourly, to_day


def fetch(ds: Dataset, days=3, workers=4, clock=time.time, air=None, traffic=None):
    """Fetch the last ``days`` days; returns ``(air by slug, traffic Hourly)``. A station that fails is left out; if everything fails an error is raised.

    ``air`` and ``traffic`` default to the real fetchers (``sources.fetch_air`` and ``sources.fetch_traffic_hourly``).
    """
    air = air or sources.fetch_air
    traffic = traffic or sources.fetch_traffic_hourly
    today = np.datetime64(int(clock()), "s").astype("datetime64[D]")
    start, end = today - (days - 1), today + 1

    def one(st):
        try:
            return st.slug, air(st.id, str(start), str(end))
        except sources.FetchError:
            return st.slug, None

    with cf.ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, ds.stations))
    got = {slug: h for slug, h in results if h is not None and h.n > 0}
    try:
        tr = traffic(days + 1)
    except sources.FetchError:
        tr = None
    if not got and tr is None:
        raise sources.FetchError("neither the air-quality nor the traffic service answered")
    return got, tr


def apply(ds: Dataset, got, tr):
    """A copy of the dataset with the fetched hours merged in."""
    air = dict(ds.air)
    for slug, h in got.items():
        air[slug] = merge_hourly(ds.air[slug], h)
    traffic = merge_hourly(ds.traffic, tr) if tr is not None else ds.traffic
    return replace(ds, air=air, traffic=traffic)


def refresh(ds: Dataset, days=3, workers=4, clock=time.time, air=None, traffic=None):
    got, tr = fetch(ds, days, workers, clock, air, traffic)
    return apply(ds, got, tr), {"stations": len(got), "traffic": tr is not None}


__all__ = ["fetch", "apply", "refresh", "to_day"]

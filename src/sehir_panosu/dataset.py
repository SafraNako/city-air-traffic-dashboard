"""The bundled data snapshot (and a refreshed copy in a user directory) as one ``Dataset``.

``SEHIR_VERI_DIZINI`` (or the ``directory`` argument) points at a directory written by ``sehir indir``; files found there shadow the
bundled ones, file by file.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .sources import POLLUTANTS
from .timeseries import Hourly

PACKAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "veri")


def _path(name, directory=None):
    user = directory or os.environ.get("SEHIR_VERI_DIZINI")
    if user and os.path.exists(os.path.join(user, name)):
        return os.path.join(user, name)
    return os.path.join(PACKAGE_DIR, name)


def data_dir(directory=None):
    return directory or os.environ.get("SEHIR_VERI_DIZINI") or PACKAGE_DIR


@dataclass
class Station:
    id: str
    name: str
    slug: str
    lat: float
    lon: float
    address: str = ""


@dataclass
class Dataset:
    stations: list
    air: dict                       # slug -> Hourly (PM10, SO2, O3, NO2, CO, aqi_<p>, aqi)
    weather: Hourly
    cams: Hourly
    traffic: Hourly                 # hourly traffic index
    traffic_daily: dict             # date, minimum, maximum, average
    holidays: list                  # ISO dates of public holidays
    coast: list
    meta: dict = field(default_factory=dict)

    def station(self, slug):
        return next(s for s in self.stations if s.slug == slug)

    def slugs(self):
        return [s.slug for s in self.stations]


def _json(name, directory=None):
    with open(_path(name, directory), encoding="utf-8") as f:
        return json.load(f)


def _npz(name, directory=None):
    with np.load(_path(name, directory), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=4)
def _load(key_paths):
    directory = key_paths[0]
    meta = _json("meta.json", directory)
    stations = [Station(s["id"], s["name"], s["slug"], s["lat"], s["lon"], s.get("address", "")) for s in _json("istasyonlar.json", directory)]
    z = _npz("hava.npz", directory)
    start, n = meta["air"]["start"], meta["air"]["hours"]
    air = {}
    for st in stations:
        cols = {k.split("|", 1)[1]: v.astype(float) for k, v in z.items() if k.startswith(st.slug + "|")}
        air[st.slug] = Hourly(start, n, cols)
    w = _npz("hava_durumu.npz", directory)
    weather = Hourly(str(w["start"]), len(next(v for k, v in w.items() if k != "start")), {k: v.astype(float) for k, v in w.items() if k != "start"})
    c = _npz("cams.npz", directory)
    cams = Hourly(str(c["start"]), len(next(v for k, v in c.items() if k != "start")), {k: v.astype(float) for k, v in c.items() if k != "start"})
    t = _npz("trafik_saatlik.npz", directory)
    traffic = Hourly(str(t["start"]), len(t["traffic"]), {"traffic": t["traffic"].astype(float)})
    with gzip.open(_path("trafik_gunluk.csv.gz", directory), "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(io.StringIO(f.read())))
    daily = {"date": np.array([r["date"] for r in rows], dtype="datetime64[D]"), **{k: np.array([float(r[k]) for r in rows]) for k in ("minimum", "maximum", "average")}}
    holidays = [h["date"] for h in _json("tatiller.json", directory)]
    return Dataset(stations, air, weather, cams, traffic, daily, holidays, _json("kiyi.json", directory), meta)


def load(directory=None):
    """The data snapshot; cached per directory."""
    directory = directory or os.environ.get("SEHIR_VERI_DIZINI") or None
    return _load((directory,))


__all__ = ["Dataset", "Station", "load", "data_dir", "POLLUTANTS", "PACKAGE_DIR"]

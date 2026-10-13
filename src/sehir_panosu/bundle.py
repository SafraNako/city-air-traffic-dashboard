"""Build a data bundle (the files ``dataset.load`` reads) from the open-data services.

Every raw service answer can be cached in a ``raw`` directory so that a build can be repeated offline; ``sehir indir`` uses it with a temporary directory.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import datetime as dt
import gzip
import io
import json
import os
import shutil

import numpy as np

from . import sources
from . import timeseries as ts
from .dataset import PACKAGE_DIR

KEEP = ["Esenler", "Aksaray", "Üsküdar 1", "Kadıköy", "Beşiktaş", "Avcılar", "Yenibosna", "Kağıthane 1", "Alibeyköy", "D-100", "Maslak", "Selimiye", "Ümraniye 1"]
DAILY_CSV_URL = "https://data.ibb.gov.tr/dataset/b3fbb6ce-03a5-4777-8c6c-111c73775523/resource/ba47eacb-a4e1-441c-ae51-0e622d4a18e2/download/traffic_index.csv"
_TR = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u", "İ": "i", "Ç": "c", "Ğ": "g", "Ö": "o", "Ş": "s", "Ü": "u"})


def slug(name):
    return name.translate(_TR).lower().strip().replace(" ", "_")


def _cached(raw, name, producer, refetch=False):
    if raw is None:
        return producer()
    path = os.path.join(raw, name)
    if os.path.exists(path) and os.path.getsize(path) > 10 and not refetch:
        with open(path, "rb") as f:
            return f.read()
    data = producer()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


def chunk_starts(start, end_next):
    """Half-year chunks on calendar boundaries covering ``[start, end_next)``: ``[(d0, d1), ...]``."""
    out, d0, stop = [], np.datetime64(start, "D"), np.datetime64(end_next, "D")
    while d0 < stop:
        y, m = int(str(d0)[:4]), int(str(d0)[5:7])
        nxt = np.datetime64(f"{y}-07-01") if m < 7 else np.datetime64(f"{y + 1}-01-01")
        d1 = min(nxt, stop)
        out.append((d0, d1))
        d0 = d1
    return out


def build(out, raw=None, start="2019-01-01", end=None, refetch=False, workers=1, progress=print, stations=KEEP, coast_source=None):
    """Write the bundle into ``out`` and return its metadata."""
    os.makedirs(out, exist_ok=True)
    end = end or dt.date.today().isoformat()
    end_next = str(np.datetime64(end, "D") + 1)
    stations_all = sources.parse_stations(_cached(raw, "stations.json", lambda: sources.http_get(sources._base("SEHIR_AIR_URL", sources.AIR_URL) + "/GetAQIStations"), refetch))
    by_name = {s["name"]: s for s in stations_all}
    chosen = [by_name[n] for n in stations]
    chunks = chunk_starts(start, end_next)

    def fetch_chunk(job):
        s, (d0, d1) = job
        body = _cached(raw, os.path.join("aq", f"{s['name'].replace(' ', '_')}_{d0}.json"),
                       lambda: json.dumps(sources.get_json(sources.air_url(s["id"], str(d0), str(d1)), timeout=300)).encode("utf-8"), refetch)
        recs = json.loads(body)
        return s["name"], d0, sources.parse_air(recs) if recs else None

    jobs = [(s, c) for s in chosen for c in chunks]
    pieces = {s["name"]: [] for s in chosen}
    with cf.ThreadPoolExecutor(max(1, workers)) as ex:
        for k, (name, d0, h) in enumerate(ex.map(fetch_chunk, jobs), 1):
            if h is not None:
                pieces[name].append(h)
            if progress and k % 10 == 0:
                progress(f"  air quality: {k}/{len(jobs)} requests")
    arrays, kept, n_hours = {}, [], None
    for s in chosen:
        sl = slug(s["name"])
        h = sources.concat_hourly(pieces[s["name"]], start, end_next)
        n_hours = h.n
        for k, v in h.arrays.items():
            arrays[f"{sl}|{k}"] = v.astype(np.float32)
        kept.append({**s, "slug": sl, "valid": {p: float(np.isfinite(h[p]).mean()) for p in sources.POLLUTANTS}})
    np.savez_compressed(os.path.join(out, "hava.npz"), **arrays)

    tr = sources.parse_traffic_hourly(_cached(raw, "traffic_hourly.json", lambda: json.dumps(sources.get_json(sources.traffic_url(sources.MAX_TRAFFIC_DAYS, "H"), timeout=300)).encode(), refetch))
    np.savez_compressed(os.path.join(out, "trafik_saatlik.npz"), traffic=tr["traffic"].astype(np.float32), start=np.array(str(tr.start)))
    daily_csv = _cached(raw, "traffic_daily.csv", lambda: sources.http_get(sources._base("SEHIR_TRAFFIC_CSV_URL", DAILY_CSV_URL), timeout=120), refetch)
    dly = sources.parse_traffic_daily(daily_csv)
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(["date", "minimum", "maximum", "average"])
    for d, lo, hi, av in zip(dly["date"], dly["minimum"], dly["maximum"], dly["average"]):
        wr.writerow([str(d), f"{lo:g}", f"{hi:g}", f"{av:.3f}"])
    with gzip.open(os.path.join(out, "trafik_gunluk.csv.gz"), "wt", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())

    w_end = str(np.datetime64(end, "D") - 6)               # the ERA5 archive lags the present by a few days
    w = sources.parse_open_meteo(_cached(raw, "weather.json", lambda: sources.http_get(sources.weather_url(start, w_end), timeout=300), refetch))
    np.savez_compressed(os.path.join(out, "hava_durumu.npz"), start=np.array(str(w.start)), **{k: v.astype(np.float32) for k, v in w.arrays.items()})
    parts = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        lo = max(np.datetime64(start, "D"), np.datetime64(f"{y}-01-01"))
        hi = min(np.datetime64(end, "D"), np.datetime64(f"{y}-12-31"))
        parts.append(sources.parse_open_meteo(_cached(raw, f"cams_{y}.json", lambda lo=lo, hi=hi: sources.http_get(sources.cams_url(str(lo), str(hi)), timeout=300), refetch)))
    cams = sources.concat_hourly(parts, start, end_next)
    np.savez_compressed(os.path.join(out, "cams.npz"), start=np.array(str(cams.start)), **{k: v.astype(np.float32) for k, v in cams.arrays.items()})
    holidays = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        holidays += sources.parse_holidays(_cached(raw, f"holidays_{y}.json", lambda y=y: sources.http_get(sources.holiday_url(y)), refetch))
    with open(os.path.join(out, "tatiller.json"), "w", encoding="utf-8") as f:
        json.dump(holidays, f, ensure_ascii=False, separators=(",", ":"))
    if coast_source:
        shutil.copyfile(coast_source, os.path.join(out, "kiyi.json"))
    elif os.path.abspath(out) != os.path.abspath(PACKAGE_DIR):
        shutil.copyfile(os.path.join(PACKAGE_DIR, "kiyi.json"), os.path.join(out, "kiyi.json"))
    with open(os.path.join(out, "istasyonlar.json"), "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=1)
    meta = {"fetched": dt.date.today().isoformat(), "air": {"start": str(ts.to_hour(start)), "hours": n_hours, "stations": len(kept)},
            "traffic_hourly": {"start": str(tr.start), "hours": tr.n}, "traffic_daily": {"first": str(dly["date"][0]), "last": str(dly["date"][-1]), "days": int(len(dly["date"]))},
            "weather": {"start": str(w.start), "hours": w.n}, "cams": {"start": str(cams.start), "hours": cams.n}, "holidays": len(holidays)}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta

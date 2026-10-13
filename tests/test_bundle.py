import json
import os

import numpy as np
import pytest

from sehir_panosu import bundle, dataset


def test_slug_transliterates_turkish_letters():
    assert bundle.slug("Üsküdar 1") == "uskudar_1" and bundle.slug("Kağıthane 1") == "kagithane_1" and bundle.slug("Beşiktaş") == "besiktas"
    assert bundle.slug("D-100 ") == "d-100" and bundle.slug("İzmit Çiğli") == "izmit_cigli" and bundle.slug("Ömer") == "omer"


def test_chunk_starts_follow_calendar_halves():
    c = bundle.chunk_starts("2019-01-01", "2020-01-02")
    assert [(str(a), str(b)) for a, b in c] == [("2019-01-01", "2019-07-01"), ("2019-07-01", "2020-01-01"), ("2020-01-01", "2020-01-02")]
    mid = bundle.chunk_starts("2019-03-15", "2019-08-01")
    assert [(str(a), str(b)) for a, b in mid] == [("2019-03-15", "2019-07-01"), ("2019-07-01", "2019-08-01")]
    assert bundle.chunk_starts("2020-02-01", "2020-02-01") == []


def test_build_from_a_real_local_server(upstream, tmp_path):
    out = str(tmp_path / "veri")
    raw = str(tmp_path / "raw")
    meta = bundle.build(out, raw=raw, start="2026-10-10", end="2026-10-12", progress=None)
    assert meta["air"]["stations"] == 13 and meta["air"]["hours"] == 72 and meta["holidays"] == 14 and meta["traffic_daily"]["days"] == 47
    assert upstream.count("/air/GetAQIStations") == 1 and upstream.count("/air/GetAQIByStationId") == 13 and upstream.count("/traffic/TrafficIndexHistory/1800/H") == 1
    assert sorted(os.listdir(out)) == sorted(os.listdir(dataset.PACKAGE_DIR))
    d = dataset.load(out)
    assert len(d.stations) == 13 and d.air["aksaray"].n == 72 and np.isfinite(d.air["aksaray"]["NO2"]).sum() > 60 and d.traffic.n > 60 and len(d.holidays) == 14
    assert json.load(open(os.path.join(out, "meta.json")))["air"]["hours"] == 72
    # a second build reads everything from the raw cache and asks the air-quality service nothing
    before = upstream.count("/air/GetAQIByStationId")
    bundle.build(out, raw=raw, start="2026-10-10", end="2026-10-12", progress=None)
    assert upstream.count("/air/GetAQIByStationId") == before
    again = bundle.build(out, raw=raw, start="2026-10-10", end="2026-10-12", progress=None, refetch=True)
    assert upstream.count("/air/GetAQIByStationId") == before + 13 and again["air"]["hours"] == 72


def test_build_reports_progress_and_uses_workers(upstream, tmp_path):
    seen = []
    bundle.build(str(tmp_path / "o"), raw=None, start="2026-01-01", end="2026-10-12", workers=3, progress=seen.append)
    assert upstream.count("/air/GetAQIByStationId") == 13 * 2 and any("requests" in m for m in seen) is (13 * 2 >= 10)


def test_build_fails_loudly_when_the_service_is_down(upstream, tmp_path):
    from sehir_panosu import sources
    sources.RETRY_DELAY = 0.0
    upstream.fail = True
    with pytest.raises(sources.FetchError):
        bundle.build(str(tmp_path / "o"), raw=None, start="2026-10-10", end="2026-10-12", progress=None)
    sources.RETRY_DELAY = 1.5

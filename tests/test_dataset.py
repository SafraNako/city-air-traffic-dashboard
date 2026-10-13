import os
import shutil

import numpy as np
import pytest

from sehir_panosu import dataset


def test_bundled_snapshot_matches_its_metadata(ds):
    m = ds.meta
    assert len(ds.stations) == m["air"]["stations"] == 13 and all(h.n == m["air"]["hours"] == 68232 for h in ds.air.values())
    assert str(ds.air["aksaray"].start) == m["air"]["start"] == "2019-01-01T00"
    assert ds.traffic.n == m["traffic_hourly"]["hours"] and str(ds.traffic.start) == m["traffic_hourly"]["start"]
    assert len(ds.traffic_daily["date"]) == m["traffic_daily"]["days"] == 3332 and str(ds.traffic_daily["date"][-1]) == m["traffic_daily"]["last"]
    assert ds.weather.n == m["weather"]["hours"] and ds.cams.n == m["cams"]["hours"] and len(ds.holidays) == m["holidays"] == 111
    assert ds.slugs()[:3] == ["esenler", "aksaray", "uskudar_1"] and ds.station("aksaray").name == "Aksaray"
    with pytest.raises(StopIteration):
        ds.station("nowhere")


def test_series_content_is_plausible(ds):
    for st in ds.stations:
        assert 40.9 < st.lat < 41.2 and 28.7 < st.lon < 29.2
        h = ds.air[st.slug]
        assert {"PM10", "SO2", "O3", "NO2", "CO", "aqi", "aqi_NO2"} <= set(h.arrays)
        assert np.isfinite(h["PM10"]).mean() > 0.9
    a = ds.air["aksaray"]
    assert np.nanmean(a["NO2"]) > 70 and a.end == np.datetime64("2026-10-14T00", "h")
    assert ds.traffic["traffic"][np.isfinite(ds.traffic["traffic"])].max() <= 100
    assert ds.weather.start == np.datetime64("2019-01-01T00", "h") and ds.cams.n == ds.air["aksaray"].n
    assert ds.holidays[0] == "2019-01-01" and "2023-04-21" in ds.holidays and "2024-04-10" in ds.holidays


def test_known_hours_are_unchanged(ds):
    a = ds.air["esenler"]
    i = a.index_of("2020-06-11T00")
    assert a.times()[i] == np.datetime64("2020-06-11T00", "h") and np.isfinite(a["PM10"][i])


def test_coast_has_land_polygons(ds):
    land = [r for r in ds.coast if not r["hole"]]
    assert len(land) >= 2 and all(len(r["ring"]) >= 3 for r in land)
    xs = np.array([p[0] for r in land for p in r["ring"]])
    ys = np.array([p[1] for r in land for p in r["ring"]])
    assert 28.1 < xs.min() < 28.5 and 29.5 < xs.max() < 29.9 and 40.5 < ys.min() < 40.8 and 41.2 < ys.max() < 41.6


def test_directory_override_and_environment(tmp_path, monkeypatch, ds):
    for f in os.listdir(dataset.PACKAGE_DIR):
        shutil.copy(os.path.join(dataset.PACKAGE_DIR, f), tmp_path / f)
    other = dataset.load(str(tmp_path))
    assert other is not ds and other.air["aksaray"].n == ds.air["aksaray"].n and np.array_equal(other.air["aksaray"]["NO2"], ds.air["aksaray"]["NO2"], equal_nan=True)
    assert dataset.load(str(tmp_path)) is other                                              # cached per directory
    monkeypatch.setenv("SEHIR_VERI_DIZINI", str(tmp_path))
    assert dataset.data_dir() == str(tmp_path) and dataset.load() is other
    monkeypatch.delenv("SEHIR_VERI_DIZINI")
    assert dataset.data_dir() == dataset.PACKAGE_DIR and dataset.data_dir("/x") == "/x"


def test_override_shadows_file_by_file(tmp_path, ds):
    import json
    meta = dict(ds.meta, fetched="1999-01-01")
    for f in ("istasyonlar.json", "hava.npz", "hava_durumu.npz", "cams.npz", "trafik_saatlik.npz", "trafik_gunluk.csv.gz", "tatiller.json", "kiyi.json"):
        shutil.copy(os.path.join(dataset.PACKAGE_DIR, f), tmp_path / f)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    assert dataset.load(str(tmp_path)).meta["fetched"] == "1999-01-01" and ds.meta["fetched"] != "1999-01-01"
    only_meta = tmp_path.parent / "only_meta"
    only_meta.mkdir()
    (only_meta / "tatiller.json").write_text(json.dumps([{"date": "2020-01-01", "name": "x"}]))
    assert dataset._json("tatiller.json", str(only_meta)) == [{"date": "2020-01-01", "name": "x"}] and dataset._json("kiyi.json", str(only_meta)) == ds.coast

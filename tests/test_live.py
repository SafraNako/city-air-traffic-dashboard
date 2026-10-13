import numpy as np
import pytest

from sehir_panosu import live, sources
from sehir_panosu import timeseries as ts

NOON = 1791892800.0                              # 2026-10-13T12:00:00 UTC


def test_fetch_asks_every_station_and_the_traffic_feed(ds, upstream):
    got, tr = live.fetch(ds, days=2, clock=lambda: NOON)
    assert set(got) == set(ds.slugs()) and tr is not None and upstream.count("/air/GetAQIByStationId") == 13
    q = [q for p, q, _ in upstream.requests if p == "/air/GetAQIByStationId"][0]
    assert q["StartDate"] == ["12.10.2026 00:00:00"] and q["EndDate"] == ["14.10.2026 00:00:00"]
    assert any(p.endswith("/TrafficIndexHistory/3/H") for p, _, _ in upstream.requests)                      # days + 1


def test_apply_merges_without_touching_the_original(ds, upstream):
    got, tr = live.fetch(ds, days=2, clock=lambda: NOON)
    new = live.apply(ds, got, tr)
    assert new is not ds and new.air["aksaray"] is not ds.air["aksaray"] and ds.air["aksaray"].n == 68232
    sample = sources.parse_air(got and open_sample())
    i = new.air["aksaray"].index_of("2026-10-12T12")
    j = sample.index_of("2026-10-12T12")
    assert np.isnan(sample["NO2"][j]) or new.air["aksaray"]["NO2"][i] == pytest.approx(sample["NO2"][j], abs=1e-9)          # the live value wins
    assert np.array_equal(new.air["aksaray"]["NO2"][:1000], ds.air["aksaray"]["NO2"][:1000], equal_nan=True)
    assert new.traffic.end >= ds.traffic.end and new.stations is ds.stations and new.coast is ds.coast


def open_sample():
    from conftest import fixture_bytes
    return fixture_bytes("air_sample.json")


def test_a_failing_station_is_left_out_but_the_rest_arrive(ds, upstream):
    upstream.fail_stations = {ds.station("aksaray").id}
    sources.RETRY_DELAY = 0.0
    try:
        got, tr = live.fetch(ds, days=2, clock=lambda: NOON)
    finally:
        sources.RETRY_DELAY = 1.5
    assert "aksaray" not in got and len(got) == 12 and tr is not None
    assert live.apply(ds, got, tr).air["aksaray"] is ds.air["aksaray"]


def test_traffic_failure_alone_is_tolerated_and_total_failure_is_an_error(ds, upstream):
    sources.RETRY_DELAY = 0.0
    try:
        upstream.fail = True
        with pytest.raises(sources.FetchError):
            live.fetch(ds, days=2, clock=lambda: NOON)

        def only_traffic(station_id, start, end):
            raise sources.FetchError("air service down")
        upstream.fail = False
        got, tr = live.fetch(ds, days=2, clock=lambda: NOON, air=only_traffic)
        assert got == {} and tr is not None
        new, info = live.refresh(ds, days=2, clock=lambda: NOON, air=only_traffic)
        assert info == {"stations": 0, "traffic": True} and new.air["aksaray"] is ds.air["aksaray"]
    finally:
        sources.RETRY_DELAY = 1.5


def test_window_arguments_reach_the_fetchers(ds):
    calls = []

    def fake_air(station_id, start, end):
        calls.append((station_id, start, end))
        return ts.Hourly("2026-10-13T00", 0)

    def fake_traffic(days):
        calls.append(("traffic", days))
        return ts.Hourly("2026-10-13T00", 1, {"traffic": [10.0]})

    live.fetch(ds, days=5, clock=lambda: NOON, air=fake_air, traffic=fake_traffic)
    assert len(calls) == 14 and ("traffic", 6) in calls and all(c[1] == "2026-10-09" and c[2] == "2026-10-14" for c in calls if c[0] != "traffic")

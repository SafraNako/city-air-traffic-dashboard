import numpy as np
import pytest

from sehir_panosu import timeseries as ts
from sehir_panosu.timeseries import Hourly


def test_to_hour_and_to_day_parse_the_service_formats():
    assert ts.to_hour("2020-06-11T05:00:00") == np.datetime64("2020-06-11T05", "h")
    assert ts.to_hour("2020-06-11 05:30") == np.datetime64("2020-06-11T05", "h")
    assert ts.to_hour("2020-06-11") == np.datetime64("2020-06-11T00", "h")
    assert ts.to_hour(np.datetime64("2020-06-11T05:45")) == np.datetime64("2020-06-11T05", "h")
    assert ts.to_day("2020-06-11T05:00:00") == np.datetime64("2020-06-11")


def test_calendar_helpers_match_known_dates():
    h = Hourly("2020-01-01T05", 100, {"a": np.arange(100.0)})                  # a Wednesday
    assert h.hour_of_day()[:3].tolist() == [5, 6, 7] and h.weekday()[0] == 2 and h.month()[0] == 1 and h.year()[0] == 2020
    assert h.end == np.datetime64("2020-01-05T09", "h") and len(h) == 100
    d = Hourly("2023-02-06T00", 48)                                           # the Monday of the Kahramanmaras earthquake
    assert d.weekday().tolist() == [0] * 24 + [1] * 24
    sat = Hourly("2020-02-29T00", 48)
    assert sat.weekday()[0] == 5 and sat.weekday()[-1] == 6 and sat.month()[0] == 2 and sat.dates()[-1] == np.datetime64("2020-03-01")


def test_set_validates_length_and_getitem():
    h = Hourly("2020-01-01T00", 5)
    h.set("x", [1, 2, 3, 4, 5])
    assert "x" in h and h["x"].tolist() == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        h.set("y", [1, 2])


def test_slice_copy_and_index():
    h = Hourly("2020-01-01T00", 48, {"x": np.arange(48.0)})
    s = h.slice("2020-01-01T10", "2020-01-02T00")
    assert s.n == 14 and s["x"][0] == 10 and s.start == np.datetime64("2020-01-01T10", "h")
    assert h.slice(None, "2020-01-01T05").n == 5 and h.slice("2019-12-01", "2030-01-01").n == 48 and h.slice("2020-01-03").n == 0
    c = h.copy()
    c["x"][0] = -1
    assert h["x"][0] == 0
    assert h.index_of("2020-01-01T07") == 7


def test_from_records_places_values_and_later_duplicates_win():
    t = np.array(["2020-01-01T02", "2020-01-01T00", "2020-01-01T02"], dtype="datetime64[h]")
    h = Hourly.from_records(t, {"v": [20.0, 5.0, 21.0]})
    assert h.n == 3 and h["v"][0] == 5.0 and np.isnan(h["v"][1]) and h["v"][2] == 21.0
    g = Hourly.from_records(t, {"v": [20.0, 5.0, 21.0]}, start="2019-12-31T22", end="2020-01-01T04")
    assert g.n == 6 and g["v"][2] == 5.0 and g["v"][4] == 21.0


def test_align_to():
    a = Hourly("2020-01-01T02", 4, {"x": [1.0, 2.0, 3.0, 4.0]})
    b = Hourly("2020-01-01T00", 5)
    out = a.align_to(b)
    assert np.isnan(out["x"][:2]).all() and out["x"][2:].tolist() == [1.0, 2.0, 3.0]


def test_rolling_mean_rules():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert np.allclose(ts.rolling_mean(x, 3, 2), [np.nan, 1.5, 1.5, 3.0, 4.5], equal_nan=True)
    assert np.allclose(ts.rolling_mean(x, 3, 3), [np.nan, np.nan, np.nan, np.nan, np.nan], equal_nan=True)
    assert np.allclose(ts.rolling_mean(x, 1, 1), x, equal_nan=True)
    big = np.arange(100.0)
    assert ts.rolling_mean(big, 24, 12)[50] == pytest.approx(np.mean(big[27:51]))


def test_daily_aggregate_and_first_midnight():
    h = Hourly("2020-01-01T05", 100, {"a": np.arange(100.0)})
    assert ts.first_midnight(h) == 19
    d, v = ts.daily(h, "a", min_count=18)
    assert d[0] == np.datetime64("2020-01-02") and v[0] == pytest.approx(np.mean(np.arange(19, 43))) and len(v) == 3
    h["a"][25] = np.nan
    assert ts.daily(h, "a", min_count=24)[1][0] != ts.daily(h, "a", min_count=24)[1][0]       # NaN: one hour missing, 24 required
    dm, vm = ts.daily(h, "a", min_count=18, how="max")
    assert vm[0] == 42.0
    mid = Hourly("2020-01-01T00", 48, {"a": np.ones(48)})
    assert ts.first_midnight(mid) == 0


def test_merge_hourly_union_and_override():
    base = Hourly("2020-01-01T00", 6, {"x": [1, 2, 3, 4, 5, 6.0], "y": [1, 1, 1, 1, 1, 1.0]})
    new = Hourly("2020-01-01T04", 5, {"x": [50, np.nan, 70, 80, 90.0]})
    out = ts.merge_hourly(base, new)
    assert out.start == base.start and out.n == 9
    assert out["x"].tolist()[:4] == [1, 2, 3, 4] and out["x"][4] == 50 and out["x"][5] == 6 and out["x"][6:].tolist() == [70, 80, 90]
    assert np.isfinite(out["y"][:6]).all() and np.isnan(out["y"][6:]).all()
    earlier = ts.merge_hourly(base, Hourly("2019-12-31T22", 3, {"x": [9, 9, 9.0]}))
    assert earlier.start == np.datetime64("2019-12-31T22", "h") and earlier["x"][:2].tolist() == [9, 9] and earlier["x"][2] == 9

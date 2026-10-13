import numpy as np
import pytest

from sehir_panosu import aqi


def test_subindex_hits_the_epa_class_limits():
    for p in aqi.POLLUTANTS:
        k = aqi.KNOTS[p]
        got = aqi.subindex(p, np.array([0.0, *k]))
        assert got == pytest.approx(aqi.LEVELS[: len(k) + 1], abs=1e-9)
    assert aqi.subindex("PM10", np.array([27.0]))[0] == pytest.approx(25.0)           # half of the first segment
    assert aqi.subindex("PM10", np.array([104.0]))[0] == pytest.approx(75.0)
    assert aqi.subindex("NO2", np.array([147.0]))[0] == pytest.approx(75.0)
    assert aqi.subindex("PM10", np.array([5000.0]))[0] == 500.0                        # capped
    out = aqi.subindex("O3", np.array([np.nan, -1.0, 54.0]))
    assert np.isnan(out[0]) and np.isnan(out[1]) and out[2] == pytest.approx(25.0)


def test_knots_are_increasing_and_identified_counts_make_sense():
    for p, k in aqi.KNOTS.items():
        assert len(k) == len(aqi.LEVELS) - 1 and all(b > a for a, b in zip(k, k[1:]))
        assert 1 <= aqi.IDENTIFIED[p] <= len(k)


def test_aggregation_windows():
    x = np.full(48, 10.0)
    x[24:] = 100.0
    pm = aqi.aggregate("PM10", x)
    assert np.isnan(pm[10]) and np.isfinite(pm[11])                                   # 11 valid hours are fewer than half of 24, 12 are enough
    assert pm[23] == pytest.approx(10.0) and pm[47] == pytest.approx(100.0) and pm[35] == pytest.approx((10 * 12 + 100 * 12) / 24)
    no2 = aqi.aggregate("NO2", x)
    assert np.array_equal(no2, x)
    o3 = aqi.aggregate("O3", x)
    assert o3[31] == pytest.approx(100.0) and o3[27] == pytest.approx((10 * 4 + 100 * 4) / 8)
    gap = x.copy()
    gap[30:40] = np.nan
    assert np.isnan(aqi.aggregate("PM10", gap)[0]) and np.isfinite(aqi.aggregate("PM10", gap)[47])


def test_half_the_window_must_be_valid():
    x = np.full(30, 50.0)
    x[:13] = np.nan                                     # the first valid hour is 13
    pm = aqi.aggregate("PM10", x)
    assert np.isnan(pm[22]) and np.isfinite(pm[24])                     # 10 valid hours up to 22, 12 up to 24


def test_overall_is_the_largest_sub_index_with_a_dominant_pollutant():
    arrays = {"PM10": np.full(48, 80.0), "NO2": np.r_[np.full(24, 50.0), np.full(24, 190.0)], "SO2": np.full(48, np.nan)}
    idx, dom, subs = aqi.aqi_series(arrays)
    assert set(subs) == {"PM10", "NO2", "SO2"}
    assert idx[30] == pytest.approx(max(subs["PM10"][30], subs["NO2"][30])) and dom[47] == "NO2" and dom[23] == "PM10"
    only_nan = aqi.aqi_series({"PM10": np.full(5, np.nan)})
    assert np.isnan(only_nan[0]).all() and only_nan[1].tolist() == [""] * 5


def test_category_names_and_boundaries():
    assert aqi.category(0)[0] == "Good" and aqi.category(50)[0] == "Good" and aqi.category(51)[0] == "Moderate" and aqi.category(100)[0] == "Moderate"
    assert aqi.category(101)[0].startswith("Unhealthy for") and aqi.category(150)[0].startswith("Unhealthy for") and aqi.category(151)[0] == "Unhealthy"
    assert aqi.category(201)[0] == "Very unhealthy" and aqi.category(301)[0] == "Hazardous" and aqi.category(900)[0] == "Hazardous"
    assert aqi.category(float("nan"))[0] == "No data"
    assert aqi.category(10)[2] == "#13a261" and aqi.category(75)[2] == "#ffcc16" and aqi.category(10)[1] == "İyi"


def test_agreement_statistics():
    pub = np.array([10.0, 20.0, 30.0, 40.0, np.nan])
    pred = np.array([10.2, 21.4, 33.0, 40.4, 50.0])
    a = aqi.agreement(pred, pub)
    assert a["n"] == 4 and a["exact"] == 0.5 and a["within1"] == 0.75 and a["within2"] == 0.75 and a["max_abs"] == 3.0
    assert np.isnan(aqi.agreement(np.array([np.nan]), np.array([1.0]))["exact"])


def test_fit_knots_recovers_planted_breakpoints():
    rng = np.random.default_rng(0)
    true = np.array([102.0, 192.0, 687.0])
    x = rng.uniform(0, 800, 30000)
    y = np.rint(np.interp(x, [0, *true], aqi.LEVELS[:4], right=150.0))
    fit, rmse = aqi.fit_knots(x, y, [95.0, 205.0, 650.0])
    assert fit == pytest.approx(true, abs=0.6) and rmse < 0.4
    fit1, _ = aqi.fit_knots(x[x < 150], y[x < 150], [90.0])
    assert fit1[0] == pytest.approx(102.0, abs=0.6)
    fit_nan, _ = aqi.fit_knots(np.r_[x, np.nan], np.r_[y, 3.0], [95.0, 205.0, 650.0])
    assert fit_nan == pytest.approx(true, abs=0.6)


def test_reconstruction_reproduces_the_published_index_on_a_real_station(ds):
    h = ds.air["aksaray"]
    idx, dom, subs = aqi.aqi_series({p: h[p] for p in aqi.POLLUTANTS})
    a = aqi.agreement(idx, h["aqi"])
    assert a["n"] > 60000 and a["within1"] > 0.995 and a["exact"] > 0.93
    for p in ("PM10", "NO2"):
        assert aqi.agreement(subs[p], h["aqi_" + p])["within1"] > 0.995
    ok = np.isfinite(idx)
    assert set(dom[ok]) <= set(aqi.POLLUTANTS)

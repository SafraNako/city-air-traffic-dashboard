import numpy as np
import pytest

from sehir_panosu import synth
from sehir_panosu import timeseries as ts


def test_diurnal_shapes():
    for kind in ("no2", "o3", "flat"):
        s = synth.diurnal_shape(kind)
        assert s.shape == (24,) and abs(s.mean()) < 1e-12
    no2 = synth.diurnal_shape("no2")
    assert int(np.argmax(no2)) in (18, 19, 20) and no2[8] > no2[12] and no2[3] < 0
    assert int(np.argmax(synth.diurnal_shape("o3"))) in (14, 15, 16)
    assert not synth.diurnal_shape("flat").any()


def test_ar1_has_the_requested_variance_and_autocorrelation():
    rng = np.random.default_rng(0)
    x = synth.ar1(200000, 0.9, 0.5, rng)
    assert x.std() == pytest.approx(0.5, rel=0.03) and np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(0.9, abs=0.01)


def test_make_series_plants_its_effects():
    rng = np.random.default_rng(1)
    h = synth.make_series(730, rng, weekend=-0.3, seasonal=0.0, sigma=0.05, phi=0.5)
    x = np.log(h["x"])
    wd = h.weekday()
    assert x[wd >= 5].mean() - x[wd < 5].mean() == pytest.approx(-0.3, abs=0.02)
    t = synth.make_series(365 * 4, rng, weekend=0, seasonal=0.0, sigma=0.02, phi=0.5, trend_per_year=0.1)
    yr = np.log(t["x"])
    assert np.polyfit(np.arange(len(yr)) / (24 * 365.25), yr, 1)[0] == pytest.approx(0.1, abs=0.01)
    cov = rng.normal(size=730 * 24)
    c = synth.make_series(730, rng, weekend=0, seasonal=0, sigma=0.01, phi=0.1, covariate=cov, covariate_beta=0.4)
    assert np.polyfit(cov, np.log(c["x"]), 1)[0] == pytest.approx(0.4, abs=0.02)


def test_plant_faults_truth_masks_are_consistent():
    rng = np.random.default_rng(2)
    x = np.round(synth.make_series(200, rng)["x"], 1)
    f, stuck, spike = synth.plant_faults(x, rng, stuck_runs=4, stuck_length=(8, 12), spikes=6)
    assert stuck.sum() >= 4 * 8 and spike.sum() == 6 and not (stuck & spike).any()
    assert np.array_equal(f[~(stuck | spike)], x[~(stuck | spike)])
    assert np.all(f[spike] > 5 * x[spike])
    runs = np.flatnonzero(np.diff(np.r_[0, stuck.astype(int), 0]) == 1)
    assert len(runs) == 4


def test_traffic_no2_follows_its_elasticity_without_noise():
    rng = np.random.default_rng(3)
    s = synth.traffic_no2(800, rng, elasticity=0.6, wind_beta=0.0, sigma=1e-9, phi=0.0)
    slope = np.polyfit(np.log(s["traffic"]), np.log(s["no2"]), 1)[0]
    assert slope == pytest.approx(0.6, abs=1e-3)
    d = synth.traffic_no2(800, rng, drop_window=(100, 119), drop_depth=0.5)
    assert d["traffic"][100:120].mean() < 0.65 * d["traffic"][:100].mean()
    wk = (d["dates"].astype(int) + 3) % 7
    assert d["traffic"][wk >= 5].mean() < d["traffic"][wk < 5].mean()
    assert len(d["dates"]) == 800 and d["truth"]["elasticity"] == 0.5 and np.all(d["wind"] > 0)


def test_series_have_the_requested_grid():
    h = synth.make_series(10, np.random.default_rng(4), start="2020-02-27T00")
    assert isinstance(h, ts.Hourly) and h.n == 240 and h.start == np.datetime64("2020-02-27T00", "h") and np.all(h["x"] > 0)

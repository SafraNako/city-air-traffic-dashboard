import math

import numpy as np
import pytest

from sehir_panosu import cycles, special, synth
from sehir_panosu import timeseries as ts
from sehir_panosu.timeseries import Hourly


def test_day_matrix_shape_and_alignment():
    h = Hourly("2020-01-01T05", 24 * 5, {"a": np.arange(24 * 5.0)})
    dates, m = cycles.day_matrix(h, "a")
    assert m.shape == (4, 24) and dates[0] == np.datetime64("2020-01-02") and m[0, 0] == 19 and m[0, 5] == 24 and m[3, 23] == 19 + 3 * 24 + 23


def test_diurnal_profile_log_and_linear():
    rng = np.random.default_rng(0)
    shape = synth.diurnal_shape("no2")
    m = np.exp(np.log(40) + shape[None, :] + rng.normal(0, 0.05, (200, 24)))
    mean, lo, hi = cycles.diurnal_profile(m, log=True)
    assert np.allclose(mean, 40 * np.exp(shape), rtol=0.03) and np.all(lo < mean) and np.all(mean < hi)
    lin, llo, lhi = cycles.diurnal_profile(m)
    assert np.all(llo < lin) and np.all(lin < lhi)
    sel, _, _ = cycles.diurnal_profile(m, np.arange(200) < 100)
    assert sel.shape == (24,)
    with pytest.raises(ValueError):
        cycles.diurnal_profile(m[:3])


def test_interval_narrows_with_more_days():
    rng = np.random.default_rng(1)
    base = np.exp(rng.normal(3.5, 0.3, (400, 24)))
    _, lo1, hi1 = cycles.diurnal_profile(base[:50], log=True, seed=1)
    _, lo2, hi2 = cycles.diurnal_profile(base, log=True, seed=1)
    assert np.mean(hi2 - lo2) < 0.6 * np.mean(hi1 - lo1)


def test_weekday_means_and_week_index_and_seasonal():
    dates = np.datetime64("2020-01-06") + np.arange(21)                     # Monday .. Sunday x 3
    daily = np.tile([1.0, 2, 3, 4, 5, 6, 7], 3)
    assert cycles.weekday_means(dates, daily).tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert cycles.week_index(dates).tolist() == [0] * 7 + [1] * 7 + [2] * 7
    assert cycles.week_index(np.datetime64("2020-01-08") + np.arange(3)).tolist() == [0, 0, 0]
    d2 = np.datetime64("2020-01-01") + np.arange(366)
    prof = cycles.seasonal_profile(d2, np.ones(366) * 2.0)
    assert prof.tolist() == [2.0] * 12


def test_weekend_effect_recovers_a_planted_effect_with_honest_intervals():
    rng = np.random.default_rng(2)
    reps, cover_a, cover_n, rej_n, effs = 60, 0, 0, 0, []
    for _ in range(reps):
        h = synth.make_series(730, rng, weekend=-0.2, sigma=0.35, phi=0.9)
        d, v = ts.daily(h, "x")
        e = cycles.weekend_effect(d, v)
        effs.append(e["effect"])
        cover_a += e["low"] <= math.exp(-0.2) <= e["high"]
        h0 = synth.make_series(730, rng, weekend=0.0, sigma=0.35, phi=0.9)
        e0 = cycles.weekend_effect(*ts.daily(h0, "x"))
        rej_n += e0["p"] < 0.05
        cover_n += e0["low"] <= 1.0 <= e0["high"]
    assert abs(np.mean(effs) + 0.211) < 0.01 and special.binom_two_sided_p(cover_a, reps, 0.95) > 0.001 and special.binom_two_sided_p(rej_n, reps, 0.05) > 0.001


def test_weekend_effect_fields_and_errors():
    rng = np.random.default_rng(3)
    d, v = ts.daily(synth.make_series(365, rng, weekend=-0.3), "x")
    e = cycles.weekend_effect(d, v)
    assert set(e) == {"effect", "ratio", "low", "high", "p", "weeks"} and e["ratio"] == pytest.approx(math.exp(e["effect"])) and e["low"] < e["ratio"] < e["high"] and e["p"] < 1e-6
    with pytest.raises(ValueError):
        cycles.weekend_effect(d[:30], v[:30])
    holes = v.copy()
    holes[::2] = np.nan
    assert cycles.weekend_effect(d, holes, min_weekdays=2, min_weeks=5)["weeks"] <= e["weeks"]


def test_holiday_effect_recovers_a_planted_drop():
    rng = np.random.default_rng(4)
    n = 1200
    dates = np.datetime64("2021-01-01") + np.arange(n)
    wd = (dates.astype(int) + 3) % 7
    base = 30.0 * np.where(wd >= 5, 0.75, 1.0) * np.exp(rng.normal(0, 0.08, n))
    cand = np.where(wd < 5)[0][20::37][:25]
    hol = [str(dates[i]) for i in cand]
    v = base.copy()
    v[cand] *= 0.6
    e, items = cycles.holiday_effect(dates, v, hol)
    assert e["ratio"] == pytest.approx(0.6, abs=0.03) and e["low"] < 0.6 < e["high"] and e["p"] < 1e-6 and e["events"] == len(items) == len(cand) == 23
    assert all(0.4 < r < 0.8 for _, r in items)
    with pytest.raises(ValueError):
        cycles.holiday_effect(dates, v, hol[:2])
    sat = [str(dates[i]) for i in np.where(wd == 5)[0][:20]]
    with pytest.raises(ValueError):
        cycles.holiday_effect(dates, v, sat)                                      # weekend holidays are not used


def test_real_weekend_effects_have_the_expected_signs(results):
    w = results.summary["cycles"]["weekend"]
    no2 = [e["ratio"] for s in w.values() for p, e in s.items() if p == "NO2" and e]
    o3 = [e["ratio"] for s in w.values() for p, e in s.items() if p == "O3" and e]
    assert len(no2) == 13 and max(no2) < 1.0 and 0.8 < np.median(no2) < 0.95
    assert len(o3) >= 8 and np.median(o3) >= 1.0
    assert all(e["low"] < e["ratio"] < e["high"] for s in w.values() for e in s.values() if e)

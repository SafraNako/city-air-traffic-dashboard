import math

import numpy as np
import pytest

from sehir_panosu import special, synth, trend
from sehir_panosu import timeseries as ts


def test_kendall_statistic_and_variance_by_hand():
    s, v = trend._kendall_s_var(np.array([1.0, 3.0, 2.0, 4.0]))
    assert s == 4.0 and v == pytest.approx(4 * 3 * 13 / 18)
    s, v = trend._kendall_s_var(np.array([1.0, 1.0, 2.0]))
    assert s == 2.0 and v == pytest.approx((3 * 2 * 11 - 2 * 1 * 9) / 18)             # one tie of size 2
    assert trend._kendall_s_var(np.array([5.0])) == (0.0, 0.0) and trend._kendall_s_var(np.array([np.nan, 1.0])) == (0.0, 0.0)


def test_seasonal_kendall_adds_the_seasons():
    table = np.tile(np.arange(6.0)[:, None], (1, 12))                         # strictly increasing in every season
    s, v, z, p = trend.seasonal_kendall(table)
    assert s == 12 * 15 and v == pytest.approx(12 * 6 * 5 * 17 / 18) and z > 5 and p < 1e-6
    flat = np.ones((6, 12))
    assert trend.seasonal_kendall(flat)[2] == 0.0 and trend.seasonal_kendall(flat)[3] == 1.0


def test_sen_slope_exact_on_a_linear_table():
    years = np.arange(8.0)[:, None]
    table = 10.0 + 2.5 * years + np.arange(12.0)[None, :]                    # same slope in every season, different levels
    assert trend.seasonal_sen_slope(table) == pytest.approx(2.5)
    table[3, 4] = np.nan
    assert trend.seasonal_sen_slope(table) == pytest.approx(2.5)
    assert np.isnan(trend.seasonal_sen_slope(np.full((1, 12), 1.0)))


def test_monthly_means_need_enough_days():
    dates = np.datetime64("2020-01-01") + np.arange(60)
    v = np.ones(60)
    v[35:50] = np.nan
    m, mm = trend.monthly_means(dates, v, min_days=20)
    assert m.tolist() == [np.datetime64("2020-01"), np.datetime64("2020-02")] and mm[0] == 1.0 and np.isnan(mm[1])
    assert trend.monthly_means(dates, v, min_days=10)[1][1] == 1.0


def test_trend_test_size_power_and_slope():
    rng = np.random.default_rng(1)
    reps, rej_null, rej_pow, slopes = 25, 0, 0, []
    for _ in range(reps):
        for planted in (0.0, -0.08):
            h = synth.make_series(365 * 7, rng, weekend=0, sigma=0.3, phi=0.85, trend_per_year=planted, seasonal=0.3)
            m, mm = trend.monthly_means(*ts.daily(h, "x"))
            t = trend.trend_test(m, mm, reps=60, seed=1)
            if planted == 0.0:
                rej_null += t["p"] < 0.05
            else:
                rej_pow += t["p"] < 0.05
                slopes.append(t["relative"])
                assert t["low"] < t["slope"] < t["high"]
    assert rej_null <= 0.15 * reps and rej_pow == reps and abs(np.mean(slopes) - (math.exp(-0.08) - 1)) < 0.015


def test_trend_test_fields_and_errors():
    rng = np.random.default_rng(2)
    h = synth.make_series(365 * 6, rng, weekend=0, trend_per_year=0.05, seasonal=0.2, sigma=0.2)
    m, mm = trend.monthly_means(*ts.daily(h, "x"))
    t = trend.trend_test(m, mm, reps=50)
    assert {"slope", "low", "high", "p", "z", "S", "years", "mean", "relative", "start_year"} <= set(t) and t["start_year"] == 2019 and t["years"] in (6, 7) and t["S"] > 0
    assert t["relative"] == pytest.approx(t["slope"] / t["mean"])
    with pytest.raises(ValueError):
        trend.trend_test(m[:20], mm[:20])


def test_annual_stats_and_exceedance_days():
    dates = np.datetime64("2020-01-01") + np.arange(366 + 365)
    v = np.full(len(dates), 10.0)
    v[:100] = np.nan                                                          # 2020 has only 266 of 366 days valid
    v[400:410] = 60.0
    a = trend.annual_stats(dates, v)
    assert a[2020]["valid_days"] == 266 and a[2020]["complete"] is False and a[2020]["mean"] == 10.0
    assert a[2021]["valid_days"] == 365 and a[2021]["complete"] and a[2021]["mean"] == pytest.approx((355 * 10 + 10 * 60) / 365)
    assert trend.exceedance_days(v, 45.0) == 10 and trend.exceedance_days(np.array([np.nan, 50.0]), 45.0) == 1


def test_real_trends_fall_for_no2(results):
    t = results.summary["trends"]
    no2 = [e["NO2"]["raw"]["relative"] for e in t.values() if "NO2" in e]
    assert len(no2) == 13 and np.median(no2) < -0.02 and sum(r < 0 for r in no2) >= 11
    sig = [e["NO2"]["raw"] for e in t.values() if "NO2" in e and e["NO2"]["raw"]["p"] < 0.05]
    assert len(sig) >= 5 and all(s["slope"] < 0 for s in sig)
    assert special.binom_two_sided_p(sum(r < 0 for r in no2), len(no2), 0.5) < 0.05

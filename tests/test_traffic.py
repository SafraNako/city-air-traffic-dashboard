import numpy as np
import pytest

from sehir_panosu import traffic
from sehir_panosu import timeseries as ts
from sehir_panosu.timeseries import Hourly


def weekly_series(days=140, seed=0, noise=0.0):
    """Hourly index with a weekday double peak, a flat weekend and a known 11 May 2020 holiday."""
    rng = np.random.default_rng(seed)
    h = Hourly("2020-03-02T00", days * 24)                                   # a Monday
    hr, wd = h.hour_of_day(), h.weekday()
    weekday = 10 + 50 * np.exp(-0.5 * ((hr - 18) / 2.0) ** 2) + 30 * np.exp(-0.5 * ((hr - 8) / 1.5) ** 2)
    weekend = 10 + 35 * np.exp(-0.5 * ((hr - 15) / 3.0) ** 2)
    x = np.where(wd < 5, weekday, weekend) * np.exp(rng.normal(0, noise, h.n))
    h.set("traffic", x)
    return h


def test_hourly_profiles_separate_day_types():
    h = weekly_series()
    hols = ["2020-03-17", "2020-03-24", "2020-04-02", "2020-04-09", "2020-04-16", "2020-04-23"]          # weekdays
    is_hol = np.isin(h.dates(), np.array(hols, dtype="datetime64[D]"))
    h["traffic"][is_hol] *= 0.4
    prof = traffic.hourly_profiles(h, "traffic", hols)
    assert set(prof) == {"weekday", "saturday", "sunday", "holiday"}
    assert int(np.argmax(prof["weekday"][0])) == 18 and int(np.argmax(prof["saturday"][0])) == 15 and prof["holiday"][0][18] < prof["weekday"][0][18] - 20
    assert all(a.shape == (24,) for a in prof["weekday"])
    assert "holiday" not in traffic.hourly_profiles(h, "traffic", [])


def test_quiet_days_find_planted_days_and_flag_nothing_else():
    rng = np.random.default_rng(1)
    n = 1500
    dates = np.datetime64("2022-01-01") + np.arange(n)
    wd = (dates.astype(int) + 3) % 7
    month = dates.astype("datetime64[M]").astype(int) % 12
    v = 30.0 * np.where(wd >= 5, 0.75, 1.0) * (1 + 0.1 * np.cos(2 * np.pi * month / 12)) * np.exp(rng.normal(0, 0.12, n))
    planted = np.arange(150, 1400, 150)
    v[planted] *= 0.45
    q = traffic.quiet_days(dates, v, [])
    flagged = set()
    for a, b, r in q["events"]:
        flagged |= set(range(int(a.astype(int)), int(b.astype(int)) + 1))
        assert r < 0.7
    truth = {int(dates[i].astype(int)) for i in planted}
    assert len(truth & flagged) >= len(truth) - 1 and len(flagged - truth) <= 2
    assert q["sd"] == pytest.approx(0.12, abs=0.03) and len(q["z"]) == n


def test_listed_holidays_are_never_reported_as_quiet_days():
    rng = np.random.default_rng(2)
    n = 900
    dates = np.datetime64("2022-01-03") + np.arange(n)
    wd = (dates.astype(int) + 3) % 7
    v = 30.0 * np.where(wd >= 5, 0.75, 1.0) * np.exp(rng.normal(0, 0.1, n))
    hol_idx = np.where(wd < 5)[0][30::40][:15]
    v[hol_idx] *= 0.4
    hols = [str(dates[i]) for i in hol_idx]
    q = traffic.quiet_days(dates, v, hols)
    flagged = {str(a) for a, b, _ in q["events"]}
    assert not (flagged & set(hols))


def test_rain_effect_recovers_a_planted_increase():
    rng = np.random.default_rng(3)
    n = 1200
    dates = np.datetime64("2021-01-01") + np.arange(n)
    wd = (dates.astype(int) + 3) % 7
    rain = np.where(rng.random(n) < 0.3, rng.uniform(1, 20, n), 0.0)
    v = 30.0 * np.where(wd >= 5, 0.75, 1.0) * np.exp(rng.normal(0, 0.1, n)) * np.where(rain >= 1, 1.1, 1.0)
    e = traffic.rain_effect(dates, v, rain, [])
    assert e["ratio"] == pytest.approx(1.1, abs=0.03) and e["p"] < 1e-6 and 300 < e["rainy_days"] < 420 and e["days"] == n
    none = traffic.rain_effect(dates, 30.0 * np.where(wd >= 5, 0.75, 1.0) * np.exp(rng.normal(0, 0.1, n)), rain, [])
    assert none["low"] < 1.0 < none["high"]


def test_typical_curve_and_now_panel():
    h = weekly_series(days=70)
    typ = traffic.typical_curve(h, "traffic", [])
    assert typ["median"].shape == (7, 24) and typ["median"][0][18] == pytest.approx(h["traffic"][18]) and np.allclose(typ["q25"], typ["q75"])
    p = traffic.now_panel(h, "traffic", [], hours=48)
    assert len(p["values"]) == 48 and p["latest"] == h["traffic"][-1] and p["percentile"] == pytest.approx(0.5) and np.allclose(p["median"], p["values"])
    h2 = h.copy()
    h2["traffic"][-1] = 500.0
    assert traffic.now_panel(h2, "traffic", [])["percentile"] == 1.0
    h3 = h.copy()
    h3["traffic"][-5:] = np.nan
    assert traffic.now_panel(h3, "traffic", [])["latest_time"] == str(h3.times()[-6])


def test_real_traffic_profiles_and_holiday_effect(results):
    t = results.summary["traffic"]
    assert t["profiles"]["weekday"]["peak_hour"] in (17, 18, 19) and t["profiles"]["sunday"]["peak_hour"] in (14, 15, 16, 17)
    assert 60 < t["profiles"]["weekday"]["peak"] < 80 and t["profiles"]["holiday"]["peak"] < t["profiles"]["weekday"]["peak"] - 15
    assert t["holiday"]["high"] < 0.8 and t["holiday"]["events"] >= 40 and t["rain"]["low"] > 1.0
    assert len(t["quiet"]) >= 15 and t["known_extended"]["detected"] == t["known_extended"]["expected"]
    assert t["source_agreement"]["r"] > 0.95


def test_synthetic_hourly_through_the_daily_pipeline():
    h = weekly_series(days=100, noise=0.05)
    dates, v = ts.daily(h, "traffic")
    wd = (dates.astype(int) + 3) % 7
    assert v[wd < 5].mean() > v[wd >= 5].mean() and len(v) == 100

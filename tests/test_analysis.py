import json

import numpy as np
import pytest

from sehir_panosu import analysis, aqi, dashboard


def test_summary_is_json_serialisable(results):
    text = json.dumps(results.summary, default=dashboard.json_default)
    back = json.loads(text)
    assert back["meta"]["air"]["stations"] == 13 and len(back["stations"]) == 13


def test_station_overview(results):
    rows = {r["slug"]: r for r in results.summary["stations"]}
    assert set(rows) == {"esenler", "aksaray", "uskudar_1", "kadikoy", "besiktas", "avcilar", "yenibosna", "kagithane_1", "alibeykoy", "d-100", "maslak", "selimiye", "umraniye_1"}
    assert max(rows, key=lambda s: rows[s]["mean"]["NO2"]) == "aksaray" and rows["aksaray"]["mean"]["NO2"] > 70 and rows["d-100"]["mean"]["PM10"] > 60
    for r in rows.values():
        assert len(r["aqi_shares"]) == len(aqi.CATEGORIES) and sum(r["aqi_shares"]) == pytest.approx(1.0) and r["aqi_shares"][0] > 0.4
        assert abs(sum(r["dominant"].values()) - 1.0) < 1e-9 and r["valid_hours"] > 50000
        assert {"NO2", "PM10"} <= set(r["annual"]) and "2022" in r["annual"]["NO2"]
    a = rows["aksaray"]["annual"]["NO2"]["2022"]
    assert a["complete"] and a["mean"] > 60 and a["days_over_who"] > 300


def test_aqi_validation_numbers(results):
    v = results.summary["aqi_validation"]
    assert v["overall"]["n"] > 870_000 and v["overall"]["within1"] > 0.996 and v["overall"]["exact"] > 0.94
    for p in aqi.POLLUTANTS:
        assert v["per_pollutant"][p]["within1"] > 0.995 and v["held_out"][p]["test"]["within1"] > 0.99
        assert all(abs(a - b) / b < 0.01 for a, b in zip(v["held_out"][p]["knots"], v["held_out"][p]["table"]))
    assert set(v["train_stations"]).isdisjoint(v["test_stations"]) and len(v["train_stations"]) == 7


def test_quality_summary(results):
    q = results.summary["quality"]
    assert q["years"][0] == 2019 and q["years"][-1] == 2026 and set(q["flags"]) == set(r["slug"] for r in results.summary["stations"])
    f = q["flags"]["maslak"]["NO2"]
    assert f["stuck"] + f["spike"] > 200 and f["run"] >= 4 and 0 < f["valid_clean"] < f["valid"]
    assert q["completeness"]["maslak"]["NO2"][2021] == 0.0 and q["completeness"]["aksaray"]["NO2"][2022] > 0.9


def test_cycles_summary(results):
    c = results.summary["cycles"]
    ak = c["profiles"]["aksaray"]
    assert ak["NO2"]["weekday"]["peak_hour"] in (19, 20, 21) and ak["NO2"]["weekday"]["min_hour"] in (5, 6, 7) and ak["O3"]["weekday"]["peak_hour"] in (14, 15, 16, 17)
    assert ak["NO2"]["weekend"]["peak"] < ak["NO2"]["weekday"]["peak"] and ak["O3"]["weekend"]["peak"] > ak["O3"]["weekday"]["peak"]
    assert len(c["seasonal"]["NO2"]) == 12 and c["seasonal"]["NO2"][0] > c["seasonal"]["NO2"][6]                 # winter above summer
    assert c["seasonal"]["O3"][6] > c["seasonal"]["O3"][0]
    assert ("aksaray", "NO2") in results.arrays["profiles"]


def test_weather_models(results):
    wm = results.summary["weather_models"]
    assert len(wm) == 13
    r2 = [e["NO2"]["r2_cv"] for e in wm.values()]
    assert np.median(r2) > 0.2 and all(e["NO2"]["r2"] > e["NO2"]["r2_cv"] for e in wm.values())
    assert all(e["NO2"]["coef"]["log_wind"][0] < 0 and e["PM10"]["coef"]["log_blh"][0] < 0 for e in wm.values() if "PM10" in e)
    dates, y, pred = results.arrays["weather_models"][("aksaray", "NO2")]
    ok = np.isfinite(y) & np.isfinite(pred)
    assert ok.sum() > 2000 and 0.4 < np.corrcoef(y[ok], pred[ok])[0, 1] < 0.9


def test_trends(results):
    t = results.summary["trends"]
    for e in t.values():
        for p, v in e.items():
            assert v["raw"]["low"] <= v["raw"]["slope"] <= v["raw"]["high"] and v["raw"]["start_year"] == 2019
    assert t["selimiye"]["NO2"]["raw"]["relative"] < -0.08 and t["alibeykoy"]["PM10"]["raw"]["relative"] < -0.1
    assert "normalised" in t["aksaray"]["NO2"] and t["aksaray"]["NO2"]["normalised"]["p"] < 0.05


def test_traffic_and_lockdown_summaries(results):
    ld = results.summary["lockdown"]
    assert ld["traffic"]["2020 spring"]["change"] < -0.5 and ld["traffic"]["2021 full lockdown"]["change"] < -0.5
    a = ld["stations"]["aksaray"]["NO2"]["2021 full lockdown"]
    assert a["change"] < -0.5 and a["p"] < 0.01 and a["z"] < -4 and a["windows"] > 2000
    pooled = ld["pooled"]["NO2"]["2021 full lockdown"]
    assert pooled["stations"] >= 9 and pooled["median_change"] < -0.1 and pooled["decreases"] >= 7 and 0 < pooled["sign_p"] <= 1
    assert set(ld["pooled"]) == {"NO2", "PM10"}


def test_episodes_and_model_summaries(results):
    e = results.summary["episodes"]
    assert 120 < e["n_episodes"] < 220 and e["episode_days"] > e["n_episodes"] and 0.6 < e["pc1_pm10"] < 0.8 and e["pc1_no2"] < e["pc1_pm10"] + 0.2
    ov = e["dust_overlap"]
    assert ov["both"] + ov["episode_only"] + ov["signal_only"] + ov["neither"] > 2500 and ov["odds_ratio"] > 5 and ov["p"] < 1e-20
    assert e["wind"]["p"] < 1e-10 and e["wind"]["sector"] in ("SE", "S", "E", "SW") and e["episodes"][0]["peak"] > 100
    assert len(results.arrays["episodes"]["median"]) == len(results.arrays["episodes"]["dates"]) == len(results.arrays["episodes"]["dust"])


def test_recent_summary(results):
    r = results.summary["recent"]
    assert len(r) == 13 and all(x["last_time"].startswith("2026-10-1") for x in r)
    assert all(0 <= x["last_aqi"] <= 300 and x["max_aqi_30d"] >= x["mean_aqi_30d"] for x in r)


def test_analysis_steps_are_deterministic(ds, prep):
    a = analysis.lockdown_effects(ds, prep)
    b = analysis.lockdown_effects(ds, prep)
    assert json.dumps(a, sort_keys=True, default=dashboard.json_default) == json.dumps(b, sort_keys=True, default=dashboard.json_default)
    s1, _ = analysis.traffic_summary(ds)
    s2, _ = analysis.traffic_summary(ds)
    assert json.dumps(s1, sort_keys=True, default=dashboard.json_default) == json.dumps(s2, sort_keys=True, default=dashboard.json_default)


def test_prepare_can_be_restricted_to_some_stations(ds):
    p = analysis.prepare(ds, {"aksaray"})
    assert set(p.clean) == {"aksaray"} and set(p.daily) == {"aksaray"} and set(p.daily["aksaray"]) == {"PM10", "NO2", "O3"}
    dates, v = p.daily["aksaray"]["NO2"]
    assert len(dates) == len(v) == 2843 and dates[0] == np.datetime64("2019-01-01")


def test_analyses_work_on_a_restricted_preparation(ds):
    p = analysis.prepare(ds, {"aksaray"})
    ld = analysis.lockdown_effects(ds, p)
    a = ld["stations"]["aksaray"]["NO2"]["2021 full lockdown"]
    assert set(ld["stations"]) == {"aksaray"} and a["rank"] == 1 and a["change"] == pytest.approx(-0.628, abs=0.01) and a["windows"] > 2400
    assert set(analysis.weather_models(ds, p)[0]) == {"aksaray"} and set(analysis.trends(ds, p)) == {"aksaray"}

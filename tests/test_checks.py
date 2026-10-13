import dataclasses

from sehir_panosu import checks, cli


def test_individual_checks_report_numbers(ds, results):
    ok, detail = checks.check_special_functions()
    assert ok and "deviation" in detail
    ok, detail = checks.check_aqi_against_published(ds, results.summary)
    assert ok and "875,599" in detail
    ok, detail = checks.check_snapshot(ds)
    assert ok and "13 stations" in detail and "negative" in detail
    ok, detail = checks.check_announced_holidays(results.summary)
    assert ok and "4 of 4" in detail
    ok, detail = checks.check_knot_recovery()
    assert ok and "recovered" in detail


def test_failures_are_reported_as_failures(ds, results):
    s = results.summary
    bad_aqi = {**s, "aqi_validation": {**s["aqi_validation"], "overall": {**s["aqi_validation"]["overall"], "within1": 0.9}}}
    assert checks.check_aqi_against_published(ds, bad_aqi)[0] is False
    bad_traffic = {**s, "traffic": {**s["traffic"], "source_agreement": {"r": 0.4, "days": 100, "mean_hourly": 10.0, "mean_csv": 30.0}}}
    assert checks.check_traffic_sources(bad_traffic)[0] is False
    bad_known = {**s, "traffic": {**s["traffic"], "known_extended": {"expected": ["a", "b"], "detected": ["a"]}}}
    assert checks.check_announced_holidays(bad_known)[0] is False
    bad_ep = {**s, "episodes": {**s["episodes"], "dust_overlap": {**s["episodes"]["dust_overlap"], "p": 0.3}}}
    assert checks.check_episodes(bad_ep)[0] is False
    bad_model = {**s, "model": {k: {p: dict(e, nmb=0.3) for p, e in v.items()} for k, v in s["model"].items()}}
    assert checks.check_model(bad_model)[0] is False
    broken = dataclasses.replace(ds, meta={**ds.meta, "air": {**ds.meta["air"], "hours": 5}})
    ok, detail = checks.check_snapshot(broken)
    assert ok is False and "hours" in detail


def test_dogrulama_runs_every_check_and_passes(capsys):
    assert cli.main(["dogrulama", "--hizli"]) == 0
    out = capsys.readouterr().out
    assert out.count("[PASS]") == 18 and "[FAIL]" not in out and "18/18 checks passed" in out
    for name in ("published AQI against the reconstruction", "announced extended holidays", "lockdown windows", "PM10 episodes against CAMS dust"):
        assert f"[PASS] {name}" in out

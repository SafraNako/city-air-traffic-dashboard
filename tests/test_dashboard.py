import dataclasses
import json
import re

import numpy as np
import pytest

from sehir_panosu import dashboard


@pytest.fixture(scope="module")
def page(dash):
    return dash.render()


def payload(page):
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_page_structure(page):
    assert page.startswith("<!doctype html>") and "<title>Istanbul Air and Traffic Dashboard</title>" in page and 'name="viewport"' in page
    for sid in ("simdi", "istasyonlar", "aqi", "kalite", "ritim", "hava", "trafik", "kilitlenme", "olaylar"):
        assert f'id="{sid}"' in page and f"href='#{sid}'" in page
    assert page.count("<img ") == 14 and page.count("data:image/png;base64,") == 14
    assert 1_000_000 < len(page) < 3_500_000


def test_no_missing_values_leak_into_the_text(page):
    visible = re.sub(r"<script.*?</script>|<style.*?</style>|data:image/png;base64,[A-Za-z0-9+/=]+", "", page, flags=re.S)
    assert "nan" not in visible.lower().replace("finance", "") and "None" not in visible and "inf %" not in visible and "nan %" not in visible


def test_numbers_on_the_page_come_from_the_analysis(page, results):
    s = results.summary
    assert f"{s['aqi_validation']['overall']['n']:,}" in page
    h = s["traffic"]["holiday"]
    assert f"{100 * (h['ratio'] - 1):+.0f} %" in page and f"{h['events']} holidays" in page
    ag = s["traffic"]["source_agreement"]
    assert f"r = {ag['r']:.3f}" in page and f"{ag['days']} days" in page
    e = s["episodes"]
    assert f"{100 * e['pc1_pm10']:.0f} % of the variance of the stations’ daily PM10" in page
    assert f"odds ratio {e['dust_overlap']['odds_ratio']:.1f}" in page
    assert "Aksaray" in page and "D-100" in page and "2024-04-08" in page and "4 of 4" in page
    assert "WHO 2021 annual NO₂ 10" in page and "above the EU annual value" in page


def test_embedded_json_matches_the_summary(page, results):
    d = payload(page)
    assert d["summary"]["aqi_validation"]["overall"]["n"] == results.summary["aqi_validation"]["overall"]["n"]
    assert d["summary"]["traffic"]["holiday"]["events"] == results.summary["traffic"]["holiday"]["events"]
    assert d["now"]["median_aqi"] > 0 and len(d["now"]["stations"]) == 13 and 0 <= d["now"]["traffic"]["percentile"] <= 1


def test_cards_show_the_headline_numbers(page):
    assert "air-quality stations" in page and "station-hours of AQI" in page and "traffic index now" in page and "median of the latest station AQI" in page


def test_the_notes_are_data_driven(dash):
    text = dash._lockdown_text()
    assert "Aksaray" in text and "2021" in text and "sign test" in text
    text = dash._exceed_text()
    assert "Aksaray" in text and "Above the WHO guideline: NO₂ at 13 of 13 stations, PM10 at 13 of 13" in text


def test_now_section_escapes_station_names(dash, ds):
    evil = [dataclasses.replace(s, name="<script>alert(1)</script> & co") if s.slug == "aksaray" else s for s in ds.stations]
    section, now = dash.now_section(dataclasses.replace(ds, stations=evil), note="a <b>note</b>")
    assert "<script>alert(1)" not in section and "&lt;script&gt;alert(1)&lt;/script&gt; &amp; co" in section and "a &lt;b&gt;note&lt;/b&gt;" in section
    assert any("<script>" in r["name"] for r in now["stations"])


def test_render_with_a_replaced_now_section(dash):
    section, now = dash.now_section(note="custom note")
    page = dash.render(section, now)
    assert "custom note" in page and page.count('id="simdi"') == 1


def test_helpers():
    assert dashboard.json_default(np.float64(1.5)) == 1.5 and dashboard.json_default(np.int64(3)) == 3 and dashboard.json_default(np.array([1, 2])) == [1, 2]
    assert dashboard.json_default(np.datetime64("2020-01-01")) == "2020-01-01"
    with pytest.raises(TypeError):
        dashboard.json_default(object())
    assert dashboard.fmt_p(0.0004) == "&lt; 0.001" and dashboard.fmt_p(0.0412) == "0.041" and dashboard.fmt_p(0.5) == "0.50" and dashboard.fmt_p(None) == "–"
    assert dashboard._share(0.999) == "more than 99 %" and dashboard._share(0.42) == "42 %" and dashboard._share(float("nan")) == "–"
    assert dashboard._pct(0.123, 1) == "12.3 %" and dashboard._pct(None) == "–"


def test_example_page_in_the_repository_is_current(page):
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "examples", "pano.html")
    if not os.path.exists(path):
        pytest.skip("examples/pano.html not generated yet")
    with open(path, encoding="utf-8") as f:
        example = f.read()
    a, b = payload(example), payload(page)
    assert a["summary"]["aqi_validation"]["overall"]["n"] == b["summary"]["aqi_validation"]["overall"]["n"] and a["summary"]["meta"] == b["summary"]["meta"]

import struct

from sehir_panosu import analysis, plots, traffic


def png_size(blob):
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", blob[16:24])


def test_every_chart_is_a_real_png(ds, results, dash):
    for name, blob in dash.charts.items():
        w, h = png_size(blob)
        assert w > 600 and h > 250 and len(blob) > 20_000, name


def test_charts_are_not_blank(dash):
    from matplotlib import image
    import io
    for key in ("map", "cycles", "traffic"):
        img = image.imread(io.BytesIO(dash.charts[key]))
        assert img.shape[2] >= 3 and (img[..., :3].min(axis=2) < 0.9).mean() > 0.03, key


def test_recent_charts(ds, results):
    rows = results.summary["recent"]
    assert png_size(plots.chart_recent(rows, ds.coast))[0] > 600
    blanks = [dict(r, last_aqi=None) for r in rows]
    assert png_size(plots.chart_recent(blanks, ds.coast))[0] > 600
    panel = traffic.now_panel(ds.traffic, "traffic", ds.holidays)
    assert png_size(plots.chart_traffic_now(panel))[0] > 600


def test_charts_survive_a_station_without_results(ds, results):
    s = results.summary
    trends = {k: v for k, v in s["trends"].items() if k != "aksaray"}
    assert png_size(plots.chart_trends(trends, s["stations"]))[0] > 600
    model = {k: v for k, v in s["model"].items() if k != "esenler"}
    assert png_size(plots.chart_model(model, s["stations"]))[0] > 600
    weekend = {k: dict(v, NO2=None) for k, v in s["cycles"]["weekend"].items()}
    assert png_size(plots.chart_weekend(weekend, s["stations"]))[0] > 600


def test_chart_helpers_use_the_official_colours():
    assert plots.aqi.CATEGORIES[0][3] == "#13a261" and plots.POLLUTANT_LABEL["NO2"] == "NO₂"
    assert analysis.LOCKDOWNS["2021 full lockdown"] == ("2021-04-29", "2021-05-17")

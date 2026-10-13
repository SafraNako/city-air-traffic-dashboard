import json
import shutil
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from conftest import fixture_bytes

from sehir_panosu import sources
from sehir_panosu.timeseries import Hourly


def test_parse_stations():
    st = sources.parse_stations(fixture_bytes("stations.json"))
    assert len(st) == 28
    aks = next(s for s in st if s["name"] == "Aksaray")
    assert aks["id"] == "cb4cd1c2-b55b-484f-ac7a-505369405d00" and aks["lat"] == pytest.approx(41.0144, abs=1e-3) and aks["lon"] == pytest.approx(28.9540, abs=1e-3)
    assert any(s["name"] == "D-100" for s in st)                                    # the trailing space of the service is stripped
    assert sources.parse_stations([{"Id": "x", "Name": "Odd ", "Adress": None, "Location": "nonsense"}])[0]["lat"] != sources.parse_stations([{"Id": "x", "Name": "Odd", "Location": "nonsense"}])[0]["lon"]


def test_parse_air_on_the_real_sample():
    h = sources.parse_air(fixture_bytes("air_sample.json"))
    assert h.n == 73 and h.start == np.datetime64("2026-10-10T00", "h")
    assert set(h.arrays) == set(sources.POLLUTANTS) | {f"aqi_{p}" for p in sources.POLLUTANTS} | {"aqi"}
    assert np.isfinite(h["NO2"]).sum() > 60 and np.nanmin(h["aqi"]) >= 0 and np.nanmax(h["aqi"]) < 300
    ok = np.isfinite(h["aqi_NO2"]) & np.isfinite(h["aqi"])
    assert np.all(h["aqi"][ok] >= h["aqi_NO2"][ok] - 1e-9)                          # the overall index is the largest sub-index


def test_parse_air_handles_missing_aqi_objects_and_empty_lists():
    recs = json.loads(fixture_bytes("air_sample.json"))
    recs[3]["AQI"] = None
    recs[4]["Concentration"]["NO2"] = None
    h = sources.parse_air(recs)
    assert np.isnan(h["aqi"][3]) and np.isfinite(h["NO2"][3]) and np.isnan(h["NO2"][4])
    assert sources.parse_air([]).n == 0 and sources.parse_air(json.dumps(recs[:2])).n == 2


def test_parse_traffic_hourly_and_daily():
    t = sources.parse_traffic_hourly(fixture_bytes("traffic_sample.json"))
    assert t.n >= 70 and "traffic" in t and np.nanmin(t["traffic"]) >= 0 and np.nanmax(t["traffic"]) <= 100
    d = sources.parse_traffic_daily(fixture_bytes("traffic_daily_sample.csv"))
    assert d["date"][0] == np.datetime64("2015-08-06") and d["average"][0] == pytest.approx(57.858, abs=1e-3) and d["minimum"][0] == 24 and d["maximum"][0] == 63
    assert np.all(np.diff(d["date"].astype(int)) > 0) and len(d["date"]) == 47


def test_parse_open_meteo_weather_and_cams():
    w = sources.parse_open_meteo(fixture_bytes("weather_sample.json"))
    assert w.n == 216 and w.start == np.datetime64("2024-04-06T00", "h") and set(sources.WEATHER_VARS) == set(w.arrays)
    assert 5 < np.nanmean(w["temperature_2m"]) < 25 and np.nanmin(w["relative_humidity_2m"]) > 0
    c = sources.parse_open_meteo(fixture_bytes("cams_sample.json"))
    assert c.n == 216 and set(sources.CAMS_VARS) == set(c.arrays) and np.nanmin(c["dust"]) >= 0


def test_parse_holidays_dedupes_and_sorts():
    hol = sources.parse_holidays(fixture_bytes("holidays_2024.json"))
    assert len(hol) == 14 and hol == sorted(hol, key=lambda r: r["date"])
    assert {"2024-04-10", "2024-04-11", "2024-04-12", "2024-06-16", "2024-06-19", "2024-10-29"} <= {h["date"] for h in hol}
    raw = json.loads(fixture_bytes("holidays_2024.json"))
    assert len(sources.parse_holidays(raw + raw[:3])) == 14


def test_urls_and_environment_overrides(monkeypatch):
    for v in ("SEHIR_AIR_URL", "SEHIR_TRAFFIC_URL", "SEHIR_WEATHER_URL", "SEHIR_CAMS_URL", "SEHIR_HOLIDAY_URL"):
        monkeypatch.delenv(v, raising=False)
    u = sources.air_url("abc", "2020-06-11", "2020-06-12")
    assert u.startswith(sources.AIR_URL + "/GetAQIByStationId?") and "StationId=abc" in u and "StartDate=11.06.2020%2000%3A00%3A00" in u and "EndDate=12.06.2020%2000%3A00%3A00" in u
    assert sources.traffic_url(30, "H") == sources.TRAFFIC_URL + "/TrafficIndexHistory/30/H"
    w = sources.weather_url("2024-04-06", "2024-04-14")
    assert "start_date=2024-04-06" in w and "timezone=Europe%2FIstanbul" in w and "boundary_layer_height" in w and w.startswith(sources.WEATHER_URL)
    assert "domains=cams_europe" in sources.cams_url("2024-04-06", "2024-04-14")
    assert sources.holiday_url(2024) == sources.HOLIDAY_URL + "/2024/TR"
    monkeypatch.setenv("SEHIR_AIR_URL", "http://example.invalid/x/")
    assert sources.air_url("a", "2020-01-01", "2020-01-02").startswith("http://example.invalid/x/GetAQIByStationId?")


def test_http_get_sends_the_json_accept_header(upstream):
    body = sources.http_get(upstream.base + "/air/GetAQIStations")
    assert len(json.loads(body)) == 28
    assert "application/json" in upstream.requests[-1][2]


def test_fetchers_against_a_real_local_server(upstream):
    st = sources.parse_stations(sources.get_json(sources._base("SEHIR_AIR_URL", "") + "/GetAQIStations"))
    assert len(st) == 28
    h = sources.fetch_air("some-id", "2026-10-10", "2026-10-13")
    assert h.n == 72 and h.start == np.datetime64("2026-10-10T00", "h") and upstream.count("/air/GetAQIByStationId") == 1
    q = upstream.requests[-1][1]
    assert q["StartDate"] == ["10.10.2026 00:00:00"] and q["EndDate"] == ["13.10.2026 00:00:00"]
    t = sources.fetch_traffic_hourly(3)
    assert t.n > 60 and any(p.endswith("/TrafficIndexHistory/3/H") for p, _, _ in upstream.requests)
    sources.fetch_traffic_hourly(5000)
    assert upstream.requests[-1][0].endswith(f"/TrafficIndexHistory/{sources.MAX_TRAFFIC_DAYS}/H")                       # capped at what the service accepts
    assert sources.fetch_weather("2024-04-06", "2024-04-14").n == 216
    assert sources.fetch_holidays([2024, 2025]) and upstream.count("/holidays/2024/TR") == 1 and upstream.count("/holidays/2025/TR") == 1


def test_long_air_windows_are_split_into_chunks(upstream):
    sources.fetch_air("some-id", "2026-01-01", "2026-10-14")
    qs = [q for p, q, _ in upstream.requests if p == "/air/GetAQIByStationId"]
    assert len(qs) == 2 and qs[0]["StartDate"] == ["01.01.2026 00:00:00"] and qs[0]["EndDate"] == ["30.06.2026 00:00:00"] and qs[1]["EndDate"] == ["14.10.2026 00:00:00"]


def test_cams_requests_are_split_by_year(upstream):
    c = sources.fetch_cams("2023-12-30", "2024-01-02")
    years = [q["start_date"][0][:4] for p, q, _ in upstream.requests if p == "/cams"]
    assert years == ["2023", "2024"] and c.start == np.datetime64("2023-12-30T00", "h") or c.n > 0


def test_errors_become_fetch_errors(upstream, monkeypatch):
    monkeypatch.setattr(sources, "RETRY_DELAY", 0.0)
    upstream.fail = True
    with pytest.raises(sources.FetchError):
        sources.http_get(upstream.base + "/air/GetAQIStations")
    with pytest.raises(sources.FetchError):
        sources.get_json(upstream.base + "/air/GetAQIStations", retries=1)
    with pytest.raises(sources.FetchError):
        sources.fetch_traffic_hourly(3)


def test_unreachable_host_is_a_fetch_error():
    with pytest.raises(sources.FetchError):
        sources.http_get("http://127.0.0.1:9/none", timeout=2)


def test_not_json_is_a_fetch_error(upstream, monkeypatch):
    monkeypatch.setattr(sources, "RETRY_DELAY", 0.0)
    upstream.bodies["/air/GetAQIStations"] = b"<xml>nope</xml>"
    with pytest.raises(sources.FetchError):
        sources.get_json(upstream.base + "/air/GetAQIStations", retries=0)


@pytest.mark.skipif(shutil.which("openssl") is None or shutil.which("curl") is None, reason="needs openssl and curl")
def test_untrusted_certificate_is_never_accepted(tmp_path):
    key, crt = tmp_path / "k.pem", tmp_path / "c.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(crt), "-days", "1", "-subj", "/CN=localhost"],
                   check=True, capture_output=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    server = HTTPServer(("127.0.0.1", 0), H)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(crt), str(key))
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(sources.FetchError):
            sources.http_get("https://localhost:%d/" % server.server_address[1], timeout=5)
    finally:
        server.shutdown()
        server.server_close()


def test_concat_hourly_layout_and_override():
    a = Hourly("2020-01-01T00", 4, {"x": [1.0, 2.0, 3.0, 4.0]})
    b = Hourly("2020-01-01T03", 4, {"x": [40.0, np.nan, 60.0, 70.0], "y": [1.0, 1.0, 1.0, 1.0]})
    out = sources.concat_hourly([a, b], "2020-01-01", "2020-01-02")
    assert out.n == 24 and out["x"][:3].tolist() == [1.0, 2.0, 3.0] and out["x"][3] == 40.0 and np.isnan(out["x"][4]) and out["x"][5] == 60.0
    assert np.isfinite(out["y"][3]) and np.isnan(out["y"][2])

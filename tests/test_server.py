import json
import threading
import urllib.error
import urllib.request

from sehir_panosu import live, server


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.status, r.read().decode("utf-8"), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8"), e.headers


class Running:
    def __init__(self, srv):
        self.srv = srv
        self.thread = threading.Thread(target=srv.serve_forever, daemon=True)

    @property
    def base(self):
        return "http://127.0.0.1:%d" % self.srv.server_address[1]

    def __enter__(self):
        self.thread.start()
        assert server.wait_until_ready(self.base + "/saglik")
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()


def test_snapshot_server_routes(dash):
    with Running(server.make_server(dash)) as s:
        status, body, headers = get(s.base + "/saglik")
        assert (status, body) == (200, "ok") and headers["Cache-Control"] == "no-store"
        status, body, headers = get(s.base + "/")
        assert status == 200 and "text/html" in headers["Content-Type"] and "Istanbul air and traffic dashboard" in body and "Live:" not in body
        status, body, _ = get(s.base + "/api/ozet")
        d = json.loads(body)
        assert status == 200 and d["stations"] == 13 and d["live_error"] is None and d["aqi_reconstruction"]["within1"] > 0.996 and d["traffic"]["holiday_ratio"] < 0.8
        assert d["mean"]["NO2"]["aksaray"] > 70 and d["episodes"]["count"] > 100
        status, body, _ = get(s.base + "/api/son")
        now = json.loads(body)
        assert status == 200 and len(now["stations"]) == 13 and 0 <= now["traffic"]["percentile"] <= 1 and now["live_error"] is None
        assert get(s.base + "/missing")[0] == 404 and get(s.base + "/saglik?x=1")[0] == 200


def test_page_is_cached(dash):
    srv = server.make_server(dash)
    with Running(srv) as s:
        a = get(s.base + "/")[1]
        b = get(s.base + "/")[1]
        assert a == b and srv.state.page is not None


def test_live_feed_refreshes_the_now_panel(dash, upstream):
    srv = server.make_server(dash, live=True, interval=3600, days=2)
    with Running(srv) as s:
        status, page, _ = get(s.base + "/")
        assert status == 200 and "Live: the last 2 days were fetched" in page and "13 stations, traffic yes" in page
        assert upstream.count("/air/GetAQIByStationId") == 13
        get(s.base + "/")
        get(s.base + "/api/son")
        assert upstream.count("/air/GetAQIByStationId") == 13                                         # cached for the interval
        assert json.loads(get(s.base + "/api/ozet")[1])["live_error"] is None


def test_live_may_be_a_custom_function(dash, upstream):
    calls = []

    def fetcher(ds, days):
        calls.append(days)
        return live.refresh(ds, days, clock=lambda: 1791892800.0)

    with Running(server.make_server(dash, live=fetcher, interval=3600, days=4)) as s:
        assert "Live:" in get(s.base + "/")[1] and calls == [4]


def test_live_failure_falls_back_to_the_snapshot(dash, upstream, monkeypatch):
    from sehir_panosu import sources
    monkeypatch.setattr(sources, "RETRY_DELAY", 0.0)
    upstream.fail = True
    with Running(server.make_server(dash, live=True, interval=3600)) as s:
        status, page, _ = get(s.base + "/")
        assert status == 200 and "Live feed unavailable" in page and "showing the data snapshot" in page
        d = json.loads(get(s.base + "/api/ozet")[1])
        assert d["live_error"] and "FetchError" in d["live_error"]
        assert json.loads(get(s.base + "/api/son")[1])["stations"]


def test_key_figures_shape(dash):
    now = {"median_aqi": 30.0, "stations": [], "traffic": {}}
    k = server.key_figures(dash, now, error="boom")
    assert k["live_error"] == "boom" and k["now"] == now and k["data_snapshot"] == "2026-10-13" and set(k["mean"]) == {"NO2", "PM10"}
    json.dumps(k)

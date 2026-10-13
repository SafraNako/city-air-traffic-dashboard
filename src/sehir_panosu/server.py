"""A small HTTP server for the dashboard, optionally refreshing the "Now" panel from the live İBB services.

Routes: ``/`` the dashboard, ``/api/ozet`` the key figures as JSON, ``/api/son`` the latest AQI per station and the traffic index as JSON, ``/saglik`` a
liveness probe. The heavy analysis is done once when the server is built; every request only recomposes the page. With ``live`` the last days are
re-fetched at most every ``interval`` seconds; if a fetch fails the page keeps the snapshot and says so.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import dashboard as dash
from . import live as livemod


class _State:
    def __init__(self, dashboard, live, interval, days):
        self.dashboard, self.live, self.interval, self.days = dashboard, live, interval, days
        self.lock = threading.Lock()
        self.fetched_at = 0.0
        self.error = None
        self.ds = dashboard.ds
        self.page = None
        self.now = None

    def current(self):
        with self.lock:
            stale = self.page is None or (self.live and time.time() - self.fetched_at > self.interval)
            if stale:
                note = None
                if self.live:
                    try:
                        fetcher = self.live if callable(self.live) else livemod.refresh
                        self.ds, info = fetcher(self.dashboard.ds, self.days)
                        self.error = None
                        note = f"Live: the last {self.days} days were fetched from the İBB services just now ({info['stations']} stations, traffic {'yes' if info['traffic'] else 'no'})."
                    except Exception as exc:                        # network down, service error: keep what we have
                        self.error = f"{type(exc).__name__}: {exc}"
                        note = f"Live feed unavailable ({self.error}); showing the data snapshot."
                        self.ds = self.dashboard.ds
                    self.fetched_at = time.time()
                section, now = self.dashboard.now_section(self.ds, note)
                self.page, self.now = self.dashboard.render(section, now), now
            return self.page, self.now


def key_figures(dashboard, now, error=None):
    s = dashboard.results.summary
    v = s["aqi_validation"]
    return {"data_snapshot": s["meta"]["fetched"], "stations": len(s["stations"]), "now": now, "live_error": error,
            "mean": {p: {r["slug"]: r["mean"][p] for r in s["stations"]} for p in ("NO2", "PM10")},
            "aqi_reconstruction": {"hours": v["overall"]["n"], "within1": v["overall"]["within1"]},
            "traffic": {"holiday_ratio": s["traffic"]["holiday"]["ratio"], "rain_ratio": s["traffic"]["rain"]["ratio"]},
            "episodes": {"count": s["episodes"]["n_episodes"], "days": s["episodes"]["episode_days"]}}


def make_server(dashboard, host="127.0.0.1", port=0, live=False, interval=300, days=3):
    """A ready (not yet serving) server; ``server.server_address`` holds the port actually bound. ``live`` may be a function ``(dataset, days) -> (dataset, info)``."""
    state = _State(dashboard, live, interval, days)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, body, content_type):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/saglik":
                self._send(200, "ok", "text/plain; charset=utf-8")
            elif path == "/":
                page, _ = state.current()
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/api/ozet":
                _, now = state.current()
                self._send(200, json.dumps(key_figures(dashboard, now, state.error), ensure_ascii=False, default=dash.json_default), "application/json; charset=utf-8")
            elif path == "/api/son":
                _, now = state.current()
                self._send(200, json.dumps({**now, "live_error": state.error}, ensure_ascii=False, default=dash.json_default), "application/json; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain; charset=utf-8")

    server = ThreadingHTTPServer((host, port), Handler)
    server.state = state
    return server


def serve(dashboard, host="127.0.0.1", port=8080, live=False, interval=300, days=3):
    server = make_server(dashboard, host, port, live, interval, days)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def wait_until_ready(url, timeout=5.0):
    """Poll ``url`` until it answers (used by tests and scripts)."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except OSError:
            time.sleep(0.05)
    return False

"""A real local HTTP server that answers like the open-data services, with saved real responses as the bodies."""
import os
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from conftest import fixture_bytes


class Upstream:
    def __init__(self):
        self.requests = []                 # (path, query dict, Accept header)
        self.fail = False
        self.fail_stations = set()         # station ids for which the air-quality route answers 500
        self.bodies = {"/air/GetAQIStations": fixture_bytes("stations.json"), "/air/GetAQIByStationId": fixture_bytes("air_sample.json"),
                       "/weather": fixture_bytes("weather_sample.json"), "/cams": fixture_bytes("cams_sample.json")}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body=b""):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                q = urllib.parse.parse_qs(parsed.query)
                outer.requests.append((parsed.path, q, self.headers.get("Accept")))
                if outer.fail:
                    return self._send(500)
                if parsed.path == "/air/GetAQIByStationId" and q.get("StationId", [""])[0] in outer.fail_stations:
                    return self._send(500)
                if re.match(r"^/traffic/TrafficIndexHistory/\d+/H$", parsed.path):
                    return self._send(200, fixture_bytes("traffic_sample.json"))
                if parsed.path == "/traffic_index.csv":
                    return self._send(200, fixture_bytes("traffic_daily_sample.csv"))
                if re.match(r"^/holidays/\d+/TR$", parsed.path):
                    return self._send(200, fixture_bytes("holidays_2024.json"))
                if parsed.path in outer.bodies:
                    return self._send(200, outer.bodies[parsed.path])
                self._send(404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.saved = {}

    @property
    def base(self):
        return "http://127.0.0.1:%d" % self.server.server_address[1]

    def __enter__(self):
        self.thread.start()
        for var, path in (("SEHIR_AIR_URL", "/air"), ("SEHIR_TRAFFIC_URL", "/traffic"), ("SEHIR_WEATHER_URL", "/weather"), ("SEHIR_CAMS_URL", "/cams"), ("SEHIR_HOLIDAY_URL", "/holidays"), ("SEHIR_TRAFFIC_CSV_URL", "/traffic_index.csv")):
            self.saved[var] = os.environ.get(var)
            os.environ[var] = self.base + path
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        for var, old in self.saved.items():
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old

    def count(self, path):
        return sum(1 for p, _, _ in self.requests if p == path)

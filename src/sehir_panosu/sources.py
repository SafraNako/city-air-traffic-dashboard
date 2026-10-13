"""Fetchers and parsers for the open data behind the dashboard.

* Istanbul Metropolitan Municipality (İBB) open data: the air-quality stations and their hourly measurements
  (concentrations and the authority's own AQI), the hourly and daily traffic index. İBB Open Data License.
* Open-Meteo: ERA5 hourly weather (archive API) and CAMS air-quality model fields. CC BY 4.0, non-commercial free use.
* Nager.Date: public holidays of Türkiye.

Every parser works on the raw service output (so tests can run on saved real responses); the fetchers add the network.
Python's urllib cannot verify the certificate chain of some of these hosts on some systems while the operating
system's trust store can, so ``http_get`` falls back to ``curl``, which keeps verification on. TLS verification is never
switched off.
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from .timeseries import HOUR, Hourly, to_day

AIR_URL = "https://api.ibb.gov.tr/havakalitesi/OpenDataPortalHandler"
TRAFFIC_URL = "https://api.ibb.gov.tr/tkmservices/api/TrafficData/v1"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
CAMS_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HOLIDAY_URL = "https://date.nager.at/api/v3/PublicHolidays"
USER_AGENT = "sehir-panosu/0.1 (open-data dashboard)"

POLLUTANTS = ("PM10", "SO2", "O3", "NO2", "CO")
CENTER = (41.01, 28.98)                    # Istanbul, the point of the weather and model series
WEATHER_VARS = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation", "surface_pressure",
                "boundary_layer_height", "shortwave_radiation")
CAMS_VARS = ("pm10", "pm2_5", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide", "dust")
MAX_AIR_DAYS = 180                         # the air-quality service fails on windows of about a year
MAX_TRAFFIC_DAYS = 1800                    # and the traffic-index history on windows beyond about 1,800 days
RETRY_DELAY = 1.5                          # seconds before the first retry, growing linearly


class FetchError(RuntimeError):
    pass


def _base(env_name, default):
    """The service URL, overridable through an environment variable (a mirror, or a local server in tests)."""
    return os.environ.get(env_name, default).rstrip("/")


def http_get(url, timeout=120):
    """GET ``url`` and return the body as bytes (urllib first, curl if the certificate chain cannot be verified)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/csv;q=0.9, */*;q=0.5"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise FetchError(f"{url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if not isinstance(reason, ssl.SSLError):
            raise FetchError(f"{url}: {exc}") from exc
        curl = shutil.which("curl")
        if curl is None:
            raise FetchError(f"{url}: certificate chain not verifiable by Python and curl is not installed ({reason}); "
                             "set SSL_CERT_FILE to a CA bundle that contains it") from exc
        done = subprocess.run([curl, "-sSL", "--fail", "-m", str(int(timeout)), "-A", USER_AGENT, "-H", "Accept: application/json, text/csv;q=0.9, */*;q=0.5", url],
                              capture_output=True)
        if done.returncode != 0:
            raise FetchError(f"{url}: curl failed ({done.stderr.decode(errors='replace').strip()})") from exc
        return done.stdout


def get_json(url, timeout=120, retries=2):
    """Fetch and decode a JSON document, retrying a few times (the İBB air-quality service times out now and then)."""
    last = None
    for attempt in range(retries + 1):
        try:
            return json.loads(http_get(url, timeout))
        except (FetchError, ValueError) as exc:
            last = exc
            time.sleep(RETRY_DELAY * (attempt + 1))
    raise FetchError(f"{url}: {last}")


# ---- air quality ------------------------------------------------------------------------------------------------------

def parse_stations(records):
    """Station dicts ``{id, name, address, lat, lon}``; the service writes the position as ``POINT (lon lat)``."""
    if isinstance(records, (bytes, str)):
        records = json.loads(records)
    out = []
    for r in records:
        loc = (r.get("Location") or "").replace("POINT", "").strip(" ()")
        try:
            lon, lat = (float(x) for x in loc.split())
        except ValueError:
            lon = lat = float("nan")
        out.append({"id": r["Id"], "name": r["Name"].strip(), "address": (r.get("Adress") or "").strip(), "lat": lat, "lon": lon})
    return out


def air_url(station_id, start, end):
    """URL of the measurements of one station between two dates (``YYYY-MM-DD``; the service wants dd.MM.yyyy HH:mm:ss)."""
    def fmt(d):
        d = to_day(d)
        y, m, dd = str(d).split("-")
        return f"{dd}.{m}.{y} 00:00:00"
    q = urllib.parse.urlencode({"StationId": station_id, "StartDate": fmt(start), "EndDate": fmt(end)}, quote_via=urllib.parse.quote)
    return _base("SEHIR_AIR_URL", AIR_URL) + "/GetAQIByStationId?" + q


def parse_air(records):
    """Hourly record list of one station to an ``Hourly``: concentrations (µg/m³) by pollutant name, the authority's sub-indices
    as ``aqi_<pollutant>`` and its overall index as ``aqi``. A record without an AQI object keeps its concentrations."""
    if isinstance(records, (bytes, str)):
        records = json.loads(records)
    if not records:
        return Hourly("2000-01-01T00", 0)
    times = np.array([r["ReadTime"][:13] for r in records], dtype="datetime64[h]")
    cols = {p: [] for p in POLLUTANTS}
    cols.update({f"aqi_{p}": [] for p in POLLUTANTS})
    cols["aqi"] = []
    for r in records:
        c = r.get("Concentration") or {}
        a = r.get("AQI") or {}
        for p in POLLUTANTS:
            cols[p].append(_f(c.get(p)))
            cols[f"aqi_{p}"].append(_f(a.get(p)))
        cols["aqi"].append(_f(a.get("AQIIndex")))
    return Hourly.from_records(times, cols)


def _f(x):
    return float("nan") if x is None else float(x)


def fetch_air(station_id, start, end, chunk_days=MAX_AIR_DAYS, workers=1, progress=None):
    """Measurements of one station between two dates, requested in chunks the service can answer; returns an ``Hourly``."""
    start, end = to_day(start), to_day(end)
    chunks, a = [], start
    while a < end:
        b = min(a + chunk_days, end)
        chunks.append((a, b))
        a = b
    parts = []
    for a, b in chunks:
        records = get_json(air_url(station_id, a, b), timeout=300)
        if records:
            parts.append(parse_air(records))
        if progress:
            progress(a, b, len(records))
    return concat_hourly(parts, start, end)


def concat_hourly(parts, start, end):
    """Place several ``Hourly`` pieces on one grid covering ``[start, end)`` (later pieces win where they overlap)."""
    lo = np.datetime64(str(to_day(start)) + "T00", "h")
    hi = np.datetime64(str(to_day(end)) + "T00", "h")
    n = int((hi - lo) / HOUR)
    out = Hourly(lo, n)
    names = []
    for p in parts:
        for k in p.arrays:
            if k not in names:
                names.append(k)
    for k in names:
        out.set(k, np.full(n, np.nan))
    for p in parts:
        shift = int((p.start - lo) / HOUR)
        a, b = max(0, shift), min(n, shift + p.n)
        if b > a:
            for k, v in p.arrays.items():
                seg = v[a - shift:b - shift]
                out.arrays[k][a:b] = np.where(np.isfinite(seg), seg, out.arrays[k][a:b])
    return out


# ---- traffic ----------------------------------------------------------------------------------------------------------

def traffic_url(days, period="H"):
    return f"{_base('SEHIR_TRAFFIC_URL', TRAFFIC_URL)}/TrafficIndexHistory/{int(days)}/{period}"


def parse_traffic_hourly(records, name="traffic"):
    """Traffic-index records (index 0-100, hourly) to an ``Hourly``."""
    if isinstance(records, (bytes, str)):
        records = json.loads(records)
    times = np.array([r["TrafficIndexDate"][:13] for r in records], dtype="datetime64[h]")
    return Hourly.from_records(times, {name: [_f(r["TrafficIndex"]) for r in records]})


def fetch_traffic_hourly(days=MAX_TRAFFIC_DAYS):
    return parse_traffic_hourly(get_json(traffic_url(min(days, MAX_TRAFFIC_DAYS), "H"), timeout=300))


def parse_traffic_daily(text):
    """The open-data CSV of the daily traffic index (``trafficindexdate, minimum, maximum, average``) to arrays."""
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    dates = np.array([r["trafficindexdate"][:10] for r in rows], dtype="datetime64[D]")
    order = np.argsort(dates, kind="stable")
    cols = {k: np.array([float(r[f"{k}_traffic_index"]) for r in rows])[order] for k in ("minimum", "maximum", "average")}
    return {"date": dates[order], **cols}


# ---- weather and model --------------------------------------------------------------------------------------------------

def open_meteo_url(base, lat, lon, start, end, variables, **extra):
    q = {"latitude": lat, "longitude": lon, "start_date": str(to_day(start)), "end_date": str(to_day(end)), "hourly": ",".join(variables),
         "timezone": "Europe/Istanbul"}
    q.update(extra)
    return base + "?" + urllib.parse.urlencode(q, safe=",")


def weather_url(start, end, lat=CENTER[0], lon=CENTER[1], variables=WEATHER_VARS):
    return open_meteo_url(_base("SEHIR_WEATHER_URL", WEATHER_URL), lat, lon, start, end, variables)


def cams_url(start, end, lat=CENTER[0], lon=CENTER[1], variables=CAMS_VARS):
    return open_meteo_url(_base("SEHIR_CAMS_URL", CAMS_URL), lat, lon, start, end, variables, domains="cams_europe")


def parse_open_meteo(doc):
    """An Open-Meteo hourly answer (local Istanbul time) to an ``Hourly`` of the requested variables."""
    if isinstance(doc, (bytes, str)):
        doc = json.loads(doc)
    h = doc["hourly"]
    times = np.array([t[:13] for t in h["time"]], dtype="datetime64[h]")
    return Hourly.from_records(times, {k: [np.nan if x is None else x for x in v] for k, v in h.items() if k != "time"})


def fetch_weather(start, end, **kw):
    return parse_open_meteo(get_json(weather_url(start, end, **kw), timeout=300))


def fetch_cams(start, end, **kw):
    """CAMS model fields; the service answers a few years at a time, so long spans are split by calendar year."""
    parts = []
    y0, y1 = int(str(to_day(start))[:4]), int(str(to_day(end))[:4])
    for y in range(y0, y1 + 1):
        a = max(to_day(start), to_day(f"{y}-01-01"))
        b = min(to_day(end), to_day(f"{y}-12-31"))
        parts.append(parse_open_meteo(get_json(cams_url(a, b, **kw), timeout=300)))
    return concat_hourly(parts, start, to_day(end) + 1)


# ---- holidays ---------------------------------------------------------------------------------------------------------

def holiday_url(year, country="TR"):
    return f"{_base('SEHIR_HOLIDAY_URL', HOLIDAY_URL)}/{int(year)}/{country}"


def parse_holidays(records):
    """Public holidays as ``{date, name, local_name}`` dicts, one per calendar day (the service may list a day twice)."""
    if isinstance(records, (bytes, str)):
        records = json.loads(records)
    seen, out = set(), []
    for r in records:
        if r["date"] in seen:
            continue
        seen.add(r["date"])
        out.append({"date": r["date"], "name": r["name"], "local_name": r.get("localName") or r["name"]})
    out.sort(key=lambda r: r["date"])
    return out


def fetch_holidays(years):
    out = []
    for y in years:
        out += parse_holidays(get_json(holiday_url(y)))
    return out

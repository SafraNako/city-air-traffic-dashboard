"""The self-contained HTML dashboard: every chart is an embedded PNG, every number comes from ``analysis``.

``build`` runs the (cached) heavy part once and returns a ``Dashboard``; ``Dashboard.render`` then composes the page in milliseconds and can swap in a
fresher "now" panel, which is what the live server does.
"""
from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field

import numpy as np

from . import analysis, aqi, dataset, plots, traffic
from .dataset import Dataset

SOURCES = [
    ("İBB open data: air-quality stations and traffic index", "https://data.ibb.gov.tr/", "İBB Open Data License; cite İBB"),
    ("Open-Meteo: ERA5 weather and CAMS air-quality model", "https://open-meteo.com/", "CC BY 4.0; non-commercial use is free"),
    ("Nager.Date: public holidays", "https://date.nager.at/", "open source, free to use"),
    ("Natural Earth: land polygons", "https://www.naturalearthdata.com/", "public domain"),
]


def json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.datetime64,)):
        return str(x)
    raise TypeError(f"{type(x)} is not JSON serialisable")


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "–"
    if p < 0.001:
        return "&lt; 0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def _esc(x):
    return html.escape(str(x))


def _t(rows, headers):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _img(png, alt):
    return f'<figure><img alt="{_esc(alt)}" src="data:image/png;base64,{base64.b64encode(png).decode()}"></figure>'


def _share(x):
    return "–" if x is None or np.isnan(x) else ("more than 99 %" if x >= 0.995 else f"{100 * x:.0f} %")


def _pct(x, digits=0):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f} %"


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1f2937;--muted:#6b7280;--line:#e5e7eb;--accent:#0072B2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{background:#12304a;color:#fff;padding:28px 16px 22px}header .in,main,footer .in{max-width:1120px;margin:0 auto}
h1{margin:0 0 6px;font-size:26px}header p{margin:4px 0;color:#cbd5e1;max-width:880px}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);z-index:5;overflow-x:auto;white-space:nowrap}
nav .in{max-width:1120px;margin:0 auto;padding:0 8px}nav a{display:inline-block;padding:10px 9px;color:var(--ink);text-decoration:none;font-size:13px}nav a:hover{color:var(--accent)}
main{padding:8px 16px 40px}section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:18px 0}
h2{margin:0 0 4px;font-size:19px}h3{margin:18px 0 6px;font-size:15px}.lead{color:var(--muted);margin:2px 0 12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:14px 0 6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.card b{display:block;font-size:24px}.card span{color:var(--muted);font-size:12.5px}
figure{margin:10px 0}figure img{max-width:100%;height:auto;display:block;margin:0 auto}
.tw{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{background:#f3f4f6;font-weight:600}tr:nth-child(even) td{background:#fafafa}
.note{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:8px 12px;font-size:13.5px;margin:10px 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:820px){.cols{grid-template-columns:1fr}}
.good{color:#047857;font-weight:600}.badc{color:#b91c1c;font-weight:600}
footer{border-top:1px solid var(--line);padding:18px 16px;color:var(--muted);font-size:13px}code{background:#f3f4f6;padding:1px 5px;border-radius:4px}
"""


@dataclass
class Dashboard:
    ds: Dataset
    results: analysis.Results
    charts: dict
    aqi_sample: tuple
    extra: dict = field(default_factory=dict)

    # ---- the part that changes with fresh data --------------------------------------------------------------------------
    def now_section(self, ds: Dataset = None, note=None):
        ds = ds or self.ds
        rows = analysis.recent_summary(ds)
        panel = traffic.now_panel(ds.traffic, "traffic", ds.holidays)
        table = []
        for r in sorted(rows, key=lambda r: -(r["last_aqi"] or 0)):
            en, tr_, color = aqi.category(r["last_aqi"] if r["last_aqi"] is not None else float("nan"))
            table.append((_esc(r["name"]), "–" if r["last_aqi"] is None else f"{r['last_aqi']:.0f}", f'<span style="color:{color};font-weight:700">●</span> {en}',
                          _esc((r["last_time"] or "").replace("T", " ")) + (":00" if r["last_time"] else ""), "–" if r["mean_aqi_30d"] is None else f"{r['mean_aqi_30d']:.0f}",
                          "–" if r["max_aqi_30d"] is None else f"{r['max_aqi_30d']:.0f}"))
        extra = f'<div class="note">{_esc(note)}</div>' if note else ""
        latest_aqi = [r["last_aqi"] for r in rows if r["last_aqi"] is not None]
        median = float(np.median(latest_aqi)) if latest_aqi else float("nan")
        page = (f'<section id="simdi"><h2>Now</h2><p class="lead">Latest hour in the data: the traffic index is {panel["latest"]:.0f} at {_esc(panel["latest_time"].replace("T", " "))}:00, '
                f'higher than {_share(panel["percentile"])} of the same weekday-hours since November 2021; the median of the stations’ latest AQI is {median:.0f}.</p>{extra}'
                + '<div class="cols"><div>' + _img(plots.chart_recent(rows, ds.coast), "latest AQI map") + "</div><div>"
                + _t(table, ["station", "AQI", "category", "last hour", "30-day mean", "30-day max"]) + "</div></div>"
                + _img(plots.chart_traffic_now(panel), "traffic index of the last hours") + "</section>")
        return page, {"stations": rows, "traffic": {"latest": panel["latest"], "latest_time": panel["latest_time"], "percentile": panel["percentile"]}, "median_aqi": median}

    # ---- static sections ----------------------------------------------------------------------------------------------------
    def _cards(self, now):
        s = self.results.summary
        meta = s["meta"]
        means = {p: np.nanmean([r["mean"][p] for r in s["stations"] if r["mean"][p] is not None]) for p in ("NO2", "PM10")}
        hours = sum(r["valid_hours"] for r in s["stations"])
        cards = [(str(len(s["stations"])), "air-quality stations"), (f"{hours / 1e6:.2f} M", "station-hours of AQI"), (f"{means['NO2']:.0f} µg/m³", "mean NO₂ over the stations"),
                 (f"{means['PM10']:.0f} µg/m³", "mean PM10 over the stations"), (f"{now['traffic']['latest']:.0f}", "traffic index now (0–100)"),
                 (f"{now['median_aqi']:.0f}", "median of the latest station AQI")]
        del meta
        return '<div class="cards">' + "".join(f"<div class='card'><b>{a}</b><span>{b}</span></div>" for a, b in cards) + "</div>"

    def _stations(self):
        s = self.results.summary
        rows = []
        for r in sorted(s["stations"], key=lambda r: -(r["mean"]["NO2"] or 0)):
            rows.append((_esc(r["name"]), *("–" if r["mean"][p] is None else f"{r['mean'][p]:.0f}" for p in ("NO2", "PM10", "O3")), "–" if r["aqi_mean"] is None else f"{r['aqi_mean']:.0f}",
                         _pct(sum(r["aqi_shares"][2:]) if r["aqi_shares"] else None, 1)))
        # annual means, complete years only
        years = sorted({int(y) for r in s["stations"] for y in r["annual"]["NO2"]})
        arows = []
        for r in s["stations"]:
            cells = []
            for y in years:
                e = r["annual"]["NO2"].get(str(y))
                n2 = "" if not e or not e["complete"] else f"{e['mean']:.0f}"
                e2 = r["annual"]["PM10"].get(str(y))
                p1 = "" if not e2 or not e2["complete"] else f"{e2['mean']:.0f}"
                cells.append(f"{n2 or '–'} / {p1 or '–'}")
            arows.append((_esc(r["name"]), *cells))
        return ('<section id="istasyonlar"><h2>The stations</h2><p class="lead">Thirteen of the municipality’s air-quality stations with data since 2019. Means are over cleaned hourly values (see data quality).</p>'
                + _img(self.charts["map"], "station map") + _t(rows, ["station", "NO₂ µg/m³", "PM10 µg/m³", "O₃ µg/m³", "mean AQI", "hours above AQI 100"])
                + '<h3>Annual means, NO₂ / PM10 (µg/m³; years with at least 75 % of the days)</h3>' + _t(arows, ["station"] + [str(y) for y in years])
                + f'<div class="note">Guidelines: WHO 2021 annual NO₂ 10, PM10 15; EU limit values (2008/50/EC) annual NO₂ 40, PM10 40 µg/m³. {self._exceed_text()}</div></section>')

    def _exceed_text(self):
        over = {"NO2": [], "PM10": []}
        above_who = {"NO2": 0, "PM10": 0}
        total = 0
        for r in self.results.summary["stations"]:
            total += 1
            for p in over:
                complete = {y: e for y, e in r["annual"][p].items() if e["complete"]}
                if complete:
                    y = max(complete)
                    if complete[y]["mean"] > analysis.EU_ANNUAL[p]:
                        over[p].append(f"{r['name']} ({complete[y]['mean']:.0f}, {y})")
                    if complete[y]["mean"] > analysis.WHO_ANNUAL[p]:
                        above_who[p] += 1
        return (f"In the latest complete year above the EU annual value: NO₂ {', '.join(over['NO2']) or 'no station'}; PM10 {', '.join(over['PM10']) or 'no station'}. "
                f"Above the WHO guideline: NO₂ at {above_who['NO2']} of {total} stations, PM10 at {above_who['PM10']} of {total}.")

    def _aqi(self):
        v = self.results.summary["aqi_validation"]
        per = v["per_pollutant"]
        rows = [(p, f"{per[p]['n']:,}", _pct(per[p]["exact"], 1), _pct(per[p]["within1"], 2), f"{per[p]['max_abs']:.0f}", _pct(v["held_out"][p]["test"]["within1"], 2),
                 ", ".join(f"{k:.0f}" for k in v["held_out"][p]["knots"]), ", ".join(f"{k:.0f}" for k in v["held_out"][p]["table"])) for p in aqi.POLLUTANTS]
        rows.append(("overall", f"{v['overall']['n']:,}", _pct(v["overall"]["exact"], 1), _pct(v["overall"]["within1"], 2), f"{v['overall']['max_abs']:.0f}", "", "", ""))
        return ('<section id="aqi"><h2>The air-quality index, reconstructed</h2><p class="lead">The municipality publishes an AQI next to the concentrations. Fitting its numbers shows how it is made: '
                'the US EPA scale with breakpoints converted to µg/m³; PM10 averaged over 24 h, O₃ and CO over 8 h, SO₂ and NO₂ hourly (at least half of the window valid); the overall index is the largest sub-index. '
                'Fitting the breakpoints on half of the stations reproduces the other half.</p>' + _img(self.charts["aqi"], "AQI categories by station")
                + _img(self.charts["aqival"], "AQI reconstruction against the published index")
                + _t(rows, ["pollutant", "hours compared", "equal after rounding", "within ±1", "largest gap", "held-out stations within ±1", "refitted knots (AQI 50, 100, 150)", "knots in the package"])
                + '<div class="note">Above the range the data reach, the breakpoints are the converted EPA values and are an assumption (for CO only the first knot, for O₃ the first two are identified). '
                  'The rare large gaps are single hours around spikes where the service and this code treat missing data differently.</div></section>')

    def _quality(self):
        q = self.results.summary["quality"]
        rows = []
        for r in self.results.summary["stations"]:
            f = q["flags"][r["slug"]]
            rows.append((_esc(r["name"]), *(f"{f[p]['stuck'] + f[p]['spike']:,} ({f[p]['run']}h)" for p in ("NO2", "PM10", "O3"))))
        return ('<section id="kalite"><h2>Data quality</h2><p class="lead">Gaps, repeated values and isolated spikes are removed before any statistics (the AQI check above uses the raw values the service published).</p>'
                + _img(self.charts["quality"], "completeness by station and year")
                + _t(rows, ["station", "NO₂ hours removed (stuck-run length)", "PM10 hours removed", "O₃ hours removed"])
                + '<div class="note">A run of identical values counts as stuck when it would occur by chance less than 0.5 times in the series (from the series’ own repeat probability). '
                  'Both rules are validated on planted faults (README, `sehir dogrulama`).</div></section>')

    def _cycles(self):
        c = self.results.summary["cycles"]
        weekend = c["weekend"]
        rows = []
        for r in self.results.summary["stations"]:
            cells = []
            for p in ("NO2", "PM10", "O3"):
                e = weekend[r["slug"]].get(p)
                cells.append("–" if not e else f"{e['ratio']:.2f} [{e['low']:.2f}, {e['high']:.2f}]")
            rows.append((_esc(r["name"]), *cells))
        return ('<section id="ritim"><h2>Daily and weekly rhythm</h2><p class="lead">NO₂ follows the rush hours, O₃ the afternoon sun; on weekends traffic-related NO₂ is lower and O₃ is not.</p>'
                + _img(self.charts["cycles"], "diurnal profiles") + '<h3>Weekend against weekday</h3>' + _img(self.charts["weekend"], "weekend ratios")
                + _t(rows, ["station", "NO₂ weekend/weekday [95 %]", "PM10", "O₃"])
                + '<div class="note">Ratios come from paired weeks (log of the weekend mean minus log of the weekday mean, Student-t interval); O₃ at or above 1 on weekends is the known “weekend effect”: '
                  'less NO fresh from traffic means less ozone is consumed.</div></section>')

    def _weather(self):
        s = self.results.summary
        wm = s["weather_models"]
        r2 = {p: [e[p]["r2_cv"] for e in wm.values() if p in e] for p in ("NO2", "PM10")}
        tr = s["trends"]
        rows = []
        for r in s["stations"]:
            t = tr.get(r["slug"], {})
            cells = []
            for p in ("NO2", "PM10"):
                e = t.get(p)
                if not e:
                    cells += ["–", "–"]
                    continue
                cells.append(f"{100 * e['raw']['relative']:+.1f} % ({fmt_p(e['raw']['p'])})")
                n = e.get("normalised")
                cells.append("–" if not n else f"{100 * n['relative']:+.1f} % ({fmt_p(n['p'])})")
            rows.append((_esc(r["name"]), *cells))
        return ('<section id="hava"><h2>Weather and trends</h2><p class="lead">Wind and a shallow boundary layer decide how much of what is emitted stays near the station. Daily log-concentration models of weather, calendar and a time trend '
                f'explain cross-validated R² of {np.nanmedian(r2["NO2"]):.2f} (NO₂) and {np.nanmedian(r2["PM10"]):.2f} (PM10) in the median (each year predicted from the others).</p>'
                + _img(self.charts["weather"], "weather elasticities and cross-validated R2") + '<h3>Trends 2019–2026</h3>' + _img(self.charts["trends"], "trends")
                + _t(rows, ["station", "NO₂ raw (p)", "NO₂ weather-normalised (p)", "PM10 raw (p)", "PM10 weather-normalised (p)"])
                + '<div class="note">Seasonal Mann–Kendall on monthly means, Theil–Sen slope as a share of the mean per year, interval by resampling years. '
                  'NO₂ fell at almost every station; part of that may be station or sensor changes, which this data cannot separate from traffic and fleet changes.</div></section>')

    def _traffic(self):
        t = self.results.summary["traffic"]
        h, rn, ag = t["holiday"], t["rain"], t["source_agreement"]
        kn = t["known_extended"]
        quiet = "".join(f"<li>{_esc(e['start'])}{'' if e['start'] == e['end'] else ' – ' + _esc(e['end'])}: {100 * e['ratio']:.0f} % of expected</li>" for e in t["quiet"])
        return ('<section id="trafik"><h2>Traffic</h2><p class="lead">The traffic index is the municipality’s congestion index (0 free-flowing … 100 gridlock), not a vehicle count.</p>'
                + _img(self.charts["traffic"], "traffic profiles and quiet days")
                + f'<div class="cols"><div><h3>What the calendar and the weather do</h3><p>Public holidays on weekdays: <b>{100 * (h["ratio"] - 1):+.0f} %</b> '
                  f'[{100 * (h["low"] - 1):+.0f}, {100 * (h["high"] - 1):+.0f}] against the same weekday in the weeks around ({h["events"]} holidays). '
                  f'Rainy days (≥ 1 mm): <b>{100 * (rn["ratio"] - 1):+.1f} %</b> [{100 * (rn["low"] - 1):+.1f}, {100 * (rn["high"] - 1):+.1f}] after weekday, month and holidays ({rn["rainy_days"]} rainy days).</p>'
                  f'<p>Two independent sources agree: the daily means of the hourly feed and the published daily file correlate at r = {ag["r"]:.3f} over {ag["days"]} days '
                  f'(means {ag["mean_hourly"]:.1f} and {ag["mean_csv"]:.1f}).</p></div>'
                  f'<div><h3>Unusually quiet days ({len(t["quiet"])} events)</h3><ul>{quiet}</ul></div></div>'
                  f'<div class="note"><b>A check against what the government announced.</b> The nine-day bayram holidays of 2024 added the weekdays 8–9 April and 20–21 June; '
                  f'the detector, which only knows weekdays, months and the public-holiday list, flagged {len(kn["detected"])} of {len(kn["expected"])} of them '
                  f'({", ".join(kn["detected"])}). It also flags the days before bayrams, election Sundays (14 and 28 May 2023, 31 March 2024) and New Year’s Eve.</div></section>')

    def _lockdown(self):
        ld = self.results.summary["lockdown"]
        rows = []
        for name, e in ld["traffic"].items():
            rows.append((_esc(name), f"{e['before']:.1f}", f"{e['index']:.1f}", f"{100 * e['change']:+.0f} %"))
        prow = []
        for p, v in ld["pooled"].items():
            for name, e in v.items():
                prow.append((p, _esc(name), e["stations"], f"{100 * e['median_change']:+.0f} %", f"{e['decreases']} of {e['stations']}", fmt_p(e["sign_p"])))
        return ('<section id="kilitlenme"><h2>The lockdowns as a natural experiment</h2><p class="lead">In spring 2020 and from 29 April to 17 May 2021 Istanbul traffic collapsed. '
                'Each station’s daily log-concentration is modelled from weather, calendar and a time trend on all days outside the windows; the mean residual inside a window is ranked among the mean residuals of all other windows of the same length.</p>'
                + _img(self.charts["lockdown"], "lockdown effects") + _t(rows, ["window", "traffic index, 4 weeks before", "in the window", "change"])
                + '<h3>Across the stations</h3>' + _t(prow, ["pollutant", "window", "stations", "median change", "stations with a decrease", "sign test p"])
                + f'<div class="note">{self._lockdown_text()}</div></section>')

    def _lockdown_text(self):
        st = self.results.summary["lockdown"]["stations"]
        names = {r["slug"]: r["name"] for r in self.results.summary["stations"]}
        best = None
        for slug, v in st.items():
            e = v.get("NO2", {}).get("2021 full lockdown")
            if e and (best is None or e["z"] < best[1]["z"]):
                best = (slug, e)
        pooled = self.results.summary["lockdown"]["pooled"]
        sentence = ""
        if best:
            sentence = f"The strongest single result is NO₂ at {names[best[0]]} in 2021 ({100 * best[1]['change']:+.0f} %, z = {best[1]['z']:.1f} against the placebo windows). "
        no2 = pooled.get("NO2", {}).get("2021 full lockdown")
        if no2:
            sentence += f"Across stations NO₂ fell at {no2['decreases']} of {no2['stations']} in 2021 (sign test p {fmt_p(no2['sign_p'])}), so the sign is consistent but not conclusive at 5 %; "
        sentence += "PM10 fell as well in 2021, so weather the model does not see may play a part. Treat the picture as supportive, not as a measured emissions effect."
        return sentence

    def _episodes(self):
        e = self.results.summary["episodes"]
        m = self.results.summary["model"]
        ov = e["dust_overlap"]
        rows = [(f"{x['start']} – {x['end']}", x["days"], f"{x['peak']:.0f}", f"{x['mean']:.0f}") for x in e["episodes"][:10]]
        nmb = {p: np.nanmedian([v[p]["nmb"] for v in m.values() if p in v]) for p in ("NO2", "PM10")}
        rr = {p: np.nanmedian([v[p]["r"] for v in m.values() if p in v]) for p in ("NO2", "PM10")}
        wind = e.get("wind")
        windtxt = "" if not wind else f" During episodes the wind blew from the {wind['sector']} ({wind['mean_direction']:.0f}°) far more often than by chance (Rayleigh p {fmt_p(wind['p'])}, {wind['days']} days)."
        return ('<section id="olaylar"><h2>Episodes and the model</h2><p class="lead">When the median PM10 of the stations is at or above 50 µg/m³ the whole city is affected, not one street: '
                f'the first principal component carries {100 * e["pc1_pm10"]:.0f} % of the variance of the stations’ daily PM10 (against {100 * (e["pc1_no2"] or 0):.0f} % for NO₂).{windtxt}</p>'
                + _img(self.charts["episodes"], "regional PM10 episodes") + _t(rows, ["episode", "days", "peak µg/m³", "mean µg/m³"])
                + f'<div class="note">The CAMS model’s dust field, which knows nothing about the stations, is a regional signal that agrees with them: on {_pct(ov["precision"])} of the days when <i>dust is high</i> (top 5 %) the stations are in an episode (odds ratio {ov["odds_ratio"]:.1f}, Fisher p {fmt_p(ov["p"])}). '
                  f'Only {_pct(ov["hit_rate"])} of the episode days have high modelled dust, so most episodes are not dust.</div>'
                + '<h3>CAMS against the stations</h3>' + _img(self.charts["model"], "model against stations")
                + f'<p class="lead">Median over stations: NO₂ normalised bias {100 * nmb["NO2"]:+.0f} %, r = {rr["NO2"]:.2f}; PM10 bias {100 * nmb["PM10"]:+.0f} %, r = {rr["PM10"]:.2f}. '
                  'A 10 km model cannot see street canyons: it underestimates NO₂ most at the traffic stations (Aksaray, D-100).</p></section>')

    def _footer(self):
        m = self.results.summary["meta"]
        src = "".join(f"<li><a href='{u}'>{n}</a>: {l}</li>" for n, u, l in SOURCES)
        return (f'<footer><div class="in"><p><b>Data</b> snapshot of {_esc(m["fetched"])}: hourly air quality of {m["air"]["stations"]} stations from {_esc(m["air"]["start"][:10])}, '
                f'traffic index hourly since {_esc(m["traffic_hourly"]["start"][:10])} and daily {_esc(m["traffic_daily"]["first"])} – {_esc(m["traffic_daily"]["last"])}, '
                f'ERA5 weather and CAMS model for central Istanbul.</p><ul>{src}</ul>'
                '<p><b>What this is not:</b> a health advisory or a regulatory assessment. The AQI is the municipality’s; official information is at havaizleme.gov.tr and ibb.istanbul.</p></div></footer>')

    def render(self, now_html=None, now=None):
        if now_html is None:
            now_html, now = self.now_section()
        nav = [("simdi", "Now"), ("istasyonlar", "Stations"), ("aqi", "AQI"), ("kalite", "Data quality"), ("ritim", "Rhythm"), ("hava", "Weather & trends"), ("trafik", "Traffic"),
               ("kilitlenme", "Lockdowns"), ("olaylar", "Episodes & model")]
        navhtml = "".join(f"<a href='#{k}'>{v}</a>" for k, v in nav)
        body = (self._cards(now) + now_html + self._stations() + self._aqi() + self._quality() + self._cycles() + self._weather() + self._traffic() + self._lockdown() + self._episodes())
        s = self.results.summary
        payload = json.dumps({"summary": {k: v for k, v in s.items() if k in ("meta", "aqi_validation", "traffic", "lockdown", "episodes")}, "now": now}, default=json_default, ensure_ascii=False)
        payload = payload.replace("</", "<\\/")
        return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Istanbul Air and Traffic Dashboard</title><style>' + CSS + '</style></head><body><header><div class="in"><h1>Istanbul air and traffic dashboard</h1>'
                '<p>Open data of the municipality (air-quality stations, traffic index) with weather and a model: the published AQI reconstructed and checked, the daily rhythm, weather-normalised trends, '
                'traffic and its holidays, the lockdowns as a natural experiment and city-wide episodes. Every number on this page is computed from the data snapshot.</p></div></header>'
                f'<nav><div class="in">{navhtml}</div></nav><main>{body}</main>{self._footer()}<script type="application/json" id="data">{payload}</script></body></html>')


def build(directory=None, results=None, ds: Dataset = None):
    """Load the data, run the analysis (unless ``results`` is given) and draw the charts."""
    ds = ds or dataset.load(directory)
    results = results or analysis.analyze(ds)
    s, a = results.summary, results.arrays
    names = {st.slug: st.name for st in ds.stations}
    mine, pub = [], []
    for st in ds.stations[:6]:
        h = ds.air[st.slug]
        idx, _, _ = aqi.aqi_series({p: h[p] for p in aqi.POLLUTANTS})
        ok = np.isfinite(idx) & np.isfinite(h["aqi"])
        mine.append(idx[ok][::7])
        pub.append(h["aqi"][ok][::7])
    sample = (np.concatenate(mine), np.concatenate(pub))
    charts = {"map": plots.map_stations(ds.coast, s["stations"]), "aqi": plots.chart_aqi(s["stations"]), "aqival": plots.chart_aqi_validation(s["aqi_validation"], sample),
              "quality": plots.chart_quality(s["quality"], s["stations"], ds.slugs()), "cycles": plots.chart_cycles(a["profiles"], ["aksaray", "avcilar"], names),
              "weekend": plots.chart_weekend(s["cycles"]["weekend"], s["stations"]), "trends": plots.chart_trends(s["trends"], s["stations"]),
              "weather": plots.chart_weather(s["weather_models"], s["stations"]), "traffic": plots.chart_traffic(a["traffic"], s["traffic"]),
              "lockdown": plots.chart_lockdown(ds, s["lockdown"], s["stations"]), "episodes": plots.chart_episodes(a["episodes"], s["episodes"]),
              "model": plots.chart_model(s["model"], s["stations"])}
    return Dashboard(ds, results, charts, sample)

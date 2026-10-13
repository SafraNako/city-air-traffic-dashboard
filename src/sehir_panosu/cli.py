"""Command line: ``sehir`` (subcommand and flag names are ASCII-transliterated Turkish)."""
from __future__ import annotations

import argparse
import sys
import tempfile
import time

import numpy as np

from . import analysis, aqi, bundle, checks, cycles, dashboard, dataset, live, server, sources, traffic, trend


def _table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _ds(args):
    return dataset.load(args.veri_dizini)


def _station(ds, name):
    key = name.lower().replace(" ", "_")
    for s in ds.stations:
        if s.slug == key or s.name.lower() == name.lower():
            return s
    raise ValueError(f"unknown station {name!r}; known: {', '.join(ds.slugs())}")


def cmd_ozet(args):
    ds = _ds(args)
    m = ds.meta
    print(f"Data snapshot of {m['fetched']}: {m['air']['stations']} stations, {m['air']['hours']:,} hourly values from {m['air']['start'][:10]}; traffic index hourly since "
          f"{m['traffic_hourly']['start'][:10]} and daily {m['traffic_daily']['first']} to {m['traffic_daily']['last']}; {m['holidays']} public holidays")
    rows = []
    for st in ds.stations:
        h = ds.air[st.slug]
        idx, _, _ = aqi.aqi_series({p: h[p] for p in aqi.POLLUTANTS})
        rows.append((st.name, f"{st.lat:.3f}", f"{st.lon:.3f}", f"{100 * np.isfinite(h['NO2']).mean():.0f}", f"{100 * np.isfinite(h['PM10']).mean():.0f}",
                     f"{np.nanmean(h['NO2']):.0f}", f"{np.nanmean(h['PM10']):.0f}", f"{np.nanmean(idx):.0f}"))
    _table(rows, ["station", "lat", "lon", "NO2 %", "PM10 %", "NO2 mean", "PM10 mean", "AQI mean"])
    return 0


def cmd_simdi(args):
    ds = _ds(args)
    if args.canli:
        try:
            ds, info = live.refresh(ds, days=args.gun)
            print(f"Live: {info['stations']} stations and the traffic feed fetched just now")
        except sources.FetchError as exc:
            print(f"live fetch failed ({exc}); showing the snapshot", file=sys.stderr)
    rows = analysis.recent_summary(ds)
    out = []
    for r in sorted(rows, key=lambda r: -(r["last_aqi"] or 0)):
        en = aqi.category(r["last_aqi"] if r["last_aqi"] is not None else float("nan"))[0]
        out.append((r["name"], "-" if r["last_aqi"] is None else f"{r['last_aqi']:.0f}", en, ((r["last_time"] or "").replace("T", " ") + ":00" if r["last_time"] else "-"), f"{r['mean_aqi_30d']:.0f}", f"{r['max_aqi_30d']:.0f}"))
    _table(out, ["station", "AQI", "category", "last hour", "30-day mean", "30-day max"])
    p = traffic.now_panel(ds.traffic, "traffic", ds.holidays)
    print(f"\nTraffic index {p['latest']:.0f} at {p['latest_time'].replace('T', ' ')}:00 (typical for this weekday-hour {p['median'][-1]:.0f}, quartiles {p['q25'][-1]:.0f}-{p['q75'][-1]:.0f}; "
          f"higher than {100 * p['percentile']:.0f} % of such hours since Nov 2021)")
    return 0


def cmd_aqi(args):
    given = {p: getattr(args, p.lower()) for p in aqi.POLLUTANTS if getattr(args, p.lower()) is not None}
    if given:
        subs = {p: float(aqi.subindex(p, np.array([v]))[0]) for p, v in given.items()}
        top = max(subs, key=subs.get)
        en, tr, _ = aqi.category(subs[top])
        print("Concentrations are taken as the averaged values the index uses (PM10 24 h, O3 and CO 8 h, SO2 and NO2 1 h).")
        _table([(p, f"{given[p]:g}", f"{subs[p]:.0f}") for p in subs], ["pollutant", "ug/m3", "sub-index"])
        print(f"\nAQI {subs[top]:.0f} ({top}): {en} / {tr}")
        return 0
    v = analysis.aqi_validation(_ds(args))
    rows = [(p, f"{e['n']:,}", f"{100 * e['exact']:.1f}", f"{100 * e['within1']:.2f}", f"{e['max_abs']:.0f}", f"{100 * v['held_out'][p]['test']['within1']:.2f}") for p, e in v["per_pollutant"].items()]
    o = v["overall"]
    rows.append(("overall", f"{o['n']:,}", f"{100 * o['exact']:.1f}", f"{100 * o['within1']:.2f}", f"{o['max_abs']:.0f}", ""))
    print("Reconstructed AQI against the published one (percent of hours):")
    _table(rows, ["pollutant", "hours", "equal", "within 1", "largest gap", "held-out within 1"])
    print("\nbreakpoints refitted on half of the stations (conc. at AQI 50, 100, ...) against the package's table:")
    for p, e in v["held_out"].items():
        print(f"  {p:5s} refitted {[round(k, 1) for k in e['knots']]}   table {[round(k, 1) for k in e['table']]}")
    return 0


def cmd_istasyon(args):
    ds = _ds(args)
    st = _station(ds, args.ad)
    prep = analysis.prepare(ds, {st.slug})
    h = ds.air[st.slug]
    print(f"{st.name} ({st.lat:.4f} N, {st.lon:.4f} E)  {st.address}")
    rows = []
    for p in analysis.CLEAN_POLLUTANTS:
        f = prep.flags[st.slug][p]
        dates, v = prep.daily[st.slug][p]
        rows.append((p, f"{100 * f['valid']:.0f}", f"{f['stuck'] + f['spike']}", f"{np.nanmean(prep.clean[st.slug][p]):.1f}" if np.isfinite(prep.clean[st.slug][p]).any() else "-",
                     f"{np.nanpercentile(v, 95):.0f}" if np.isfinite(v).any() else "-"))
    _table(rows, ["pollutant", "valid %", "hours removed", "mean (cleaned)", "95th pct of daily means"])
    idx, dom, _ = aqi.aqi_series({p: h[p] for p in aqi.POLLUTANTS})
    ok = np.isfinite(idx)
    print("\nshare of hours by AQI category:")
    for _, name, _, _ in aqi.CATEGORIES:
        print(f"  {name:32s} {100 * np.mean(np.array([aqi.category(v)[0] for v in idx[ok]]) == name):5.1f} %")
    print("\nannual means (years with >= 75 % of the days):")
    for p in ("NO2", "PM10"):
        dates, v = prep.daily[st.slug][p]
        ann = trend.annual_stats(dates, v)
        print("  " + p + ": " + ", ".join(f"{y}: {e['mean']:.0f}" for y, e in ann.items() if e["complete"]))
    return 0


def cmd_ritim(args):
    ds = _ds(args)
    st = _station(ds, args.istasyon)
    prep = analysis.prepare(ds, {st.slug})
    p = args.kirletici
    h = prep.clean[st.slug]
    dates, m = cycles.day_matrix(h, p)
    wd = (dates.astype(int) + 3) % 7
    prof = {k: cycles.diurnal_profile(m, mask, reps=150, log=True) for k, mask in (("weekday", wd < 5), ("weekend", wd >= 5))}
    print(f"{st.name}, {p}: hourly geometric mean (ug/m3) with 95 % bootstrap interval by day")
    _table([(f"{hr:02d}", *(f"{prof[k][0][hr]:.1f} [{prof[k][1][hr]:.1f}, {prof[k][2][hr]:.1f}]" for k in ("weekday", "weekend"))) for hr in range(24)], ["hour", "weekday", "weekend"])
    d, v = prep.daily[st.slug][p]
    e = cycles.weekend_effect(d, v)
    print(f"\nweekend / weekday = {e['ratio']:.3f}  95 % interval [{e['low']:.3f}, {e['high']:.3f}]  p = {_p(e['p'])}  ({e['weeks']} paired weeks)")
    return 0


def cmd_trend(args):
    ds = _ds(args)
    st = _station(ds, args.istasyon)
    prep = analysis.prepare(ds, {st.slug})
    p = args.kirletici
    dates, v = prep.daily[st.slug][p]
    months, mm = trend.monthly_means(dates, v)
    t = trend.trend_test(months, mm, reps=args.tekrar)
    print(f"{st.name}, {p}: seasonal Mann-Kendall z = {t['z']:.2f} (p = {_p(t['p'])}); Theil-Sen {t['slope']:+.2f} ug/m3 per year "
          f"[{t['low']:+.2f}, {t['high']:+.2f}] = {100 * t['relative']:+.1f} % of the mean per year; {t['years']} years from {t['start_year']}")
    return 0


def cmd_trafik(args):
    ds = _ds(args)
    s, _ = analysis.traffic_summary(ds)
    print("Traffic index (0-100, congestion):")
    _table([(k, f"{e['peak_hour']:02d}:00", f"{e['peak']:.0f}", f"{e['night_min']:.0f}") for k, e in s["profiles"].items()], ["day type", "peak hour", "peak", "night minimum"])
    h, r, a = s["holiday"], s["rain"], s["source_agreement"]
    print(f"\nweekday public holidays: {100 * (h['ratio'] - 1):+.0f} % [{100 * (h['low'] - 1):+.0f}, {100 * (h['high'] - 1):+.0f}] ({h['events']} holidays)")
    print(f"rainy days (>= 1 mm): {100 * (r['ratio'] - 1):+.1f} % [{100 * (r['low'] - 1):+.1f}, {100 * (r['high'] - 1):+.1f}] ({r['rainy_days']} rainy days)")
    print(f"hourly feed against the published daily file: r = {a['r']:.3f} over {a['days']} days")
    print(f"\n{len(s['quiet'])} unusually quiet days or runs:")
    for e in s["quiet"][: args.adet]:
        print(f"  {e['start']}{'' if e['start'] == e['end'] else ' .. ' + e['end']}  {100 * e['ratio']:.0f} % of expected")
    print(f"announced extra holiday weekdays of 2024 flagged: {len(s['known_extended']['detected'])} of {len(s['known_extended']['expected'])}")
    return 0


def cmd_kilitlenme(args):
    ds = _ds(args)
    prep = analysis.prepare(ds)
    ld = analysis.lockdown_effects(ds, prep)
    print("Traffic index in the windows:")
    _table([(k, f"{e['before']:.1f}", f"{e['index']:.1f}", f"{100 * e['change']:+.0f} %") for k, e in ld["traffic"].items()], ["window", "4 weeks before", "in window", "change"])
    rows = []
    for slug, v in ld["stations"].items():
        for p, w in v.items():
            for name, e in w.items():
                rows.append((slug, p, name, f"{100 * e['change']:+.0f} %", f"{e['z']:+.1f}", _p(e["p"])))
    print("\nWeather-normalised change against placebo windows:")
    _table(rows, ["station", "pollutant", "window", "change", "z", "placebo p"])
    print("\nAcross stations:")
    _table([(p, n, e["stations"], f"{100 * e['median_change']:+.0f} %", f"{e['decreases']}/{e['stations']}", _p(e["sign_p"])) for p, v in ld["pooled"].items() for n, e in v.items()],
           ["pollutant", "window", "stations", "median change", "decreases", "sign test p"])
    return 0


def cmd_olay(args):
    ds = _ds(args)
    prep = analysis.prepare(ds)
    cams = analysis.cams_daily(ds)
    e, _ = analysis.episodes_summary(ds, prep, cams)
    print(f"{e['n_episodes']} regional PM10 episodes (median of stations >= 50 ug/m3), {e['episode_days']} days; first principal component of PM10 {100 * e['pc1_pm10']:.0f} % of the variance"
          + ("" if e["pc1_no2"] is None else f" (NO2: {100 * e['pc1_no2']:.0f} %)"))
    ov = e["dust_overlap"]
    print(f"CAMS dust (top 5 %, > {e['dust_threshold']:.0f} ug/m3) against episodes: both {ov['both']}, episode only {ov['episode_only']}, dust only {ov['signal_only']}, neither {ov['neither']}; "
          f"odds ratio {ov['odds_ratio']:.1f}, Fisher p {_p(ov['p'])}")
    if "wind" in e:
        w = e["wind"]
        print(f"wind during episodes: from the {w['sector']} ({w['mean_direction']:.0f} deg), R = {w['r']:.2f}, Rayleigh p {_p(w['p'])} ({w['days']} days)")
    _table([(f"{x['start']} .. {x['end']}", x["days"], f"{x['peak']:.0f}", f"{x['mean']:.0f}") for x in e["episodes"][: args.adet]], ["episode", "days", "peak", "mean"])
    return 0


def cmd_model(args):
    ds = _ds(args)
    prep = analysis.prepare(ds)
    m = analysis.model_comparison(ds, prep, analysis.cams_daily(ds))
    rows = []
    for st in ds.stations:
        for p, e in m.get(st.slug, {}).items():
            rows.append((st.name, p, f"{e['n']}", f"{e['obs_mean']:.1f}", f"{e['mod_mean']:.1f}", f"{100 * e['nmb']:+.0f} %", f"{e['rmse']:.1f}", f"{e['r']:.2f} [{e['r_low']:.2f}, {e['r_high']:.2f}]"))
    _table(rows, ["station", "pollutant", "days", "station mean", "CAMS mean", "norm. bias", "RMSE", "r [95 %]"])
    return 0


def cmd_indir(args):
    raw = args.ham or tempfile.mkdtemp(prefix="sehir_ham_")
    print(f"Downloading {args.baslangic} .. {args.bitis or 'today'} into {args.cikti} (raw answers cached in {raw}; the air-quality service answers slowly, be patient)", flush=True)
    meta = bundle.build(args.cikti, raw=raw, start=args.baslangic, end=args.bitis, workers=args.paralel, progress=lambda m: print(m, flush=True))
    print(f"Done: {meta['air']['stations']} stations, {meta['air']['hours']:,} hours. Use it with:  SEHIR_VERI_DIZINI={args.cikti} sehir pano")
    return 0


def cmd_pano(args):
    t0 = time.time()
    dash = dashboard.build(args.veri_dizini)
    page = dash.render()
    with open(args.cikti, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"{args.cikti}: {len(page) / 1024:.0f} KB in {time.time() - t0:.0f} s")
    return 0


def cmd_sunucu(args):
    print("Preparing the dashboard (the analysis takes about 40 s) ...", flush=True)
    dash = dashboard.build(args.veri_dizini)
    print(f"Serving on http://{args.adres}:{args.port}/  ({'live feed every %d s' % args.aralik if args.canli else 'snapshot only'}); Ctrl-C to stop", flush=True)
    server.serve(dash, args.adres, args.port, live=args.canli, interval=args.aralik)
    return 0


def cmd_dogrulama(args):
    t0 = time.time()
    results = checks.run_all(quick=args.hizli, ds=_ds(args))
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed in {time.time() - t0:.0f} s")
    return 1 if failed else 0


def build_parser():
    p = argparse.ArgumentParser(prog="sehir", description="Istanbul air quality and traffic from open data: statistics, validation and a dashboard.")
    p.add_argument("--veri-dizini", default=None, help="directory written by 'sehir indir' (default: bundled snapshot or $SEHIR_VERI_DIZINI)")
    sub = p.add_subparsers(dest="komut", required=True)
    sub.add_parser("ozet", help="data snapshot summary").set_defaults(fn=cmd_ozet)
    s = sub.add_parser("simdi", help="latest AQI per station and the traffic index")
    s.add_argument("--canli", action="store_true", help="fetch the last days from the live services")
    s.add_argument("--gun", type=int, default=3)
    s.set_defaults(fn=cmd_simdi)
    s = sub.add_parser("aqi", help="check the AQI reconstruction, or compute an AQI from concentrations")
    for pol in aqi.POLLUTANTS:
        s.add_argument(f"--{pol.lower()}", type=float, default=None, help=f"averaged {pol} in ug/m3")
    s.set_defaults(fn=cmd_aqi)
    s = sub.add_parser("istasyon", help="one station in detail")
    s.add_argument("--ad", required=True, help="station name or slug, e.g. aksaray")
    s.set_defaults(fn=cmd_istasyon)
    s = sub.add_parser("ritim", help="daily profile and weekend effect of one station and pollutant")
    s.add_argument("--istasyon", required=True)
    s.add_argument("--kirletici", choices=list(analysis.CLEAN_POLLUTANTS), default="NO2")
    s.set_defaults(fn=cmd_ritim)
    s = sub.add_parser("trend", help="seasonal Mann-Kendall trend of one station and pollutant")
    s.add_argument("--istasyon", required=True)
    s.add_argument("--kirletici", choices=["NO2", "PM10", "O3"], default="NO2")
    s.add_argument("--tekrar", type=int, default=300, help="bootstrap repetitions")
    s.set_defaults(fn=cmd_trend)
    s = sub.add_parser("trafik", help="traffic profiles, holiday and rain effects, quiet days")
    s.add_argument("--adet", type=int, default=15)
    s.set_defaults(fn=cmd_trafik)
    sub.add_parser("kilitlenme", help="weather-normalised change in the lockdown windows").set_defaults(fn=cmd_kilitlenme)
    s = sub.add_parser("olay", help="regional PM10 episodes and their link to modelled dust")
    s.add_argument("--adet", type=int, default=10)
    s.set_defaults(fn=cmd_olay)
    sub.add_parser("model", help="CAMS model against the stations").set_defaults(fn=cmd_model)
    s = sub.add_parser("indir", help="refresh the data from the live services")
    s.add_argument("--cikti", default="sehir_veri")
    s.add_argument("--ham", default=None, help="directory that caches the raw service answers")
    s.add_argument("--baslangic", default="2019-01-01")
    s.add_argument("--bitis", default=None)
    s.add_argument("--paralel", type=int, default=3)
    s.set_defaults(fn=cmd_indir)
    s = sub.add_parser("pano", help="write the HTML dashboard")
    s.add_argument("--cikti", default="pano.html")
    s.set_defaults(fn=cmd_pano)
    s = sub.add_parser("sunucu", help="serve the dashboard over HTTP")
    s.add_argument("--adres", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--canli", action="store_true", help="refresh the Now panel from the live services")
    s.add_argument("--aralik", type=int, default=300, help="seconds between live fetches")
    s.set_defaults(fn=cmd_sunucu)
    s = sub.add_parser("dogrulama", help="run the validation checks")
    s.add_argument("--hizli", action="store_true", help="fewer repetitions")
    s.set_defaults(fn=cmd_dogrulama)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (sources.FetchError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())


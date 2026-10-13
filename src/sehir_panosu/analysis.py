"""The full analysis behind the dashboard: one pass over the data snapshot that returns plain numbers (JSON-able) and the arrays the charts need."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import aqi, cycles, episodes, model, normalize as nz, quality, special, traffic, trend, weather
from .dataset import Dataset
from .timeseries import Hourly, daily

# WHO 2021 air-quality guidelines (24 h, µg/m³) and the annual values; EU limit values of Directive 2008/50/EC
WHO_DAILY = {"PM10": 45.0, "NO2": 25.0}
WHO_ANNUAL = {"PM10": 15.0, "NO2": 10.0}
EU_ANNUAL = {"PM10": 40.0, "NO2": 40.0}
CLEAN_POLLUTANTS = ("PM10", "NO2", "O3")
LOCKDOWNS = {"2020 spring": ("2020-03-23", "2020-04-30"), "2021 full lockdown": ("2021-04-29", "2021-05-17")}
# extended (nine-day) bayram holidays announced for 2024, used to check the detection of unusually quiet days
KNOWN_EXTENDED = ("2024-04-08", "2024-04-09", "2024-06-20", "2024-06-21")


@dataclass
class Results:
    summary: dict
    arrays: dict = field(default_factory=dict)


@dataclass
class Prep:
    clean: dict            # slug -> Hourly with cleaned PM10, NO2, O3
    daily: dict            # slug -> {pollutant: (dates, values)}
    flags: dict            # slug -> {pollutant: {stuck, spike, impossible, run}}
    wdates: np.ndarray
    wv: dict


def prepare(ds: Dataset, slugs=None):
    """Clean the series and build the daily means (for ``slugs`` only, or all stations)."""
    clean, dly, flags = {}, {}, {}
    for st in ds.stations:
        if slugs is not None and st.slug not in slugs:
            continue
        h = ds.air[st.slug]
        out = Hourly(h.start, h.n)
        dly[st.slug], flags[st.slug] = {}, {}
        for p in CLEAN_POLLUTANTS:
            c, fl, run = quality.clean(p, h[p])
            out.set(p, c)
            flags[st.slug][p] = {"stuck": int(fl["stuck"].sum()), "spike": int(fl["spike"].sum()), "impossible": int(fl["impossible"].sum()), "run": int(run),
                                 "valid": float(np.isfinite(h[p]).mean()), "valid_clean": float(np.isfinite(c).mean())}
            dly[st.slug][p] = daily(out, p, 18)
        clean[st.slug] = out
    wdates, wv = weather.daily_weather(ds.weather)
    return Prep(clean, dly, flags, wdates, wv)


def _prepared(ds, prep):
    """The stations that ``prep`` was built for (all of them unless ``prepare`` was restricted)."""
    return [st for st in ds.stations if st.slug in prep.daily]


def _model_fit(dates, y, extra, holidays, train=None, lags=14, trend_term=True):
    """Daily log-concentration regression on calendar, weather and (optionally) a linear time trend that absorbs level changes of the station."""
    extra = dict(extra)
    if trend_term:
        extra["years"] = (dates - dates[0]).astype(float) / 365.25
    X, names = nz.build_design(dates, holidays, extra)
    f, use = nz.fit_complete_rows(X, y, train=train, lags=lags)
    return f, X, names


def station_overview(ds, prep):
    rows = []
    cats = {}
    for st in ds.stations:
        h = ds.air[st.slug]
        raw = {p: h[p] for p in aqi.POLLUTANTS}
        idx, dom, subs = aqi.aqi_series(raw)
        ok = np.isfinite(idx)
        shares = [float(np.mean(np.array([aqi.category(v)[0] for v in idx[ok]]) == name)) for _, name, _, _ in aqi.CATEGORIES] if ok.any() else []
        dom_counts = {p: float(np.mean(dom[ok] == p)) for p in aqi.POLLUTANTS}
        years = h.year()
        annual = {}
        for p in ("NO2", "PM10"):
            dates, v = prep.daily[st.slug][p]
            ann = trend.annual_stats(dates, v)
            annual[p] = {str(y): ann[y] | {"days_over_who": trend.exceedance_days(v[dates.astype("datetime64[Y]").astype(int) + 1970 == y], WHO_DAILY[p])} for y in ann}
        means = {p: float(np.nanmean(prep.clean[st.slug][p])) if np.isfinite(prep.clean[st.slug][p]).any() else None for p in CLEAN_POLLUTANTS}
        rows.append({"slug": st.slug, "name": st.name, "lat": st.lat, "lon": st.lon, "mean": means, "aqi_mean": float(np.nanmean(idx[ok])) if ok.any() else None,
                     "aqi_shares": shares, "dominant": dom_counts, "annual": annual, "valid_hours": int(ok.sum())})
        cats[st.slug] = (years, idx)
    return rows


def aqi_validation(ds):
    """How well the reconstructed AQI reproduces the published one; also a held-out test of the fitted breakpoints (fit on half of the stations, test on the rest)."""
    sub = {p: ([], []) for p in aqi.POLLUTANTS}
    overall_pred, overall_pub = [], []
    for st in ds.stations:
        h = ds.air[st.slug]
        idx, _, subs = aqi.aqi_series({p: h[p] for p in aqi.POLLUTANTS})
        overall_pred.append(idx)
        overall_pub.append(h["aqi"])
        for p in aqi.POLLUTANTS:
            sub[p][0].append(subs[p])
            sub[p][1].append(h["aqi_" + p])
    per = {p: aqi.agreement(np.concatenate(sub[p][0]), np.concatenate(sub[p][1])) for p in aqi.POLLUTANTS}
    total = aqi.agreement(np.concatenate(overall_pred), np.concatenate(overall_pub))
    # held-out breakpoints: fit on the first half of the stations, apply to the others
    slugs = ds.slugs()
    train, test = slugs[::2], slugs[1::2]
    held = {}
    start_guess = {"PM10": [54, 154, 254], "NO2": [102, 192, 690], "SO2": [92, 199, 490], "CO": [5190], "O3": [108, 250]}
    for p in aqi.POLLUTANTS:
        xs, ys, xt, yt = [], [], [], []
        for group, X, Y in ((train, xs, ys), (test, xt, yt)):
            for s in group:
                h = ds.air[s]
                X.append(aqi.aggregate(p, h[p]))
                Y.append(h["aqi_" + p])
        x, y = np.concatenate(xs), np.concatenate(ys)
        sel = np.isfinite(x) & np.isfinite(y)
        # fit on a thinned sample (every 5th hour) to keep the search quick
        knots, rmse = aqi.fit_knots(x[sel][::5], y[sel][::5], start_guess[p])
        pred = aqi.subindex(p, np.concatenate(xt), knots)
        a = aqi.agreement(pred, np.concatenate(yt))
        held[p] = {"knots": [float(k) for k in knots], "table": [float(k) for k in aqi.KNOTS[p][:len(knots)]], "train_rmse": rmse, "test": a}
    return {"per_pollutant": per, "overall": total, "held_out": held, "train_stations": train, "test_stations": test}


def quality_summary(ds, prep):
    years = np.unique(ds.air[ds.stations[0].slug].year())
    comp = {}
    for st in ds.stations:
        h = ds.air[st.slug]
        comp[st.slug] = {p: quality.completeness_by_year(h.year(), h[p]) for p in ("NO2", "PM10")}
    return {"flags": prep.flags, "completeness": comp, "years": [int(y) for y in years]}


def cycles_summary(ds, prep):
    out = {"profiles": {}, "weekend": {}, "seasonal": {}}
    arrays = {}
    for st in ds.stations:
        out["weekend"][st.slug] = {}
        for p in CLEAN_POLLUTANTS:
            dates, v = prep.daily[st.slug][p]
            try:
                e = cycles.weekend_effect(dates, v)
                out["weekend"][st.slug][p] = e
            except ValueError:
                out["weekend"][st.slug][p] = None
    for slug in ("aksaray", "esenler", "avcilar", "besiktas"):
        h = prep.clean[slug]
        out["profiles"][slug] = {}
        for p in CLEAN_POLLUTANTS:
            if not np.isfinite(h[p]).any():
                continue
            dates, m = cycles.day_matrix(h, p)
            wd = (dates.astype(int) + 3) % 7
            prof = {}
            for label, mask in (("weekday", wd < 5), ("weekend", wd >= 5)):
                if mask.sum() > 20 and np.isfinite(m[mask]).sum() > 24 * 20:
                    prof[label] = cycles.diurnal_profile(m, mask, reps=120, log=True)
            if prof:
                arrays[(slug, p)] = prof
                out["profiles"][slug][p] = {k: {"peak_hour": int(np.nanargmax(v[0])), "peak": float(np.nanmax(v[0])), "min_hour": int(np.nanargmin(v[0])), "min": float(np.nanmin(v[0]))} for k, v in prof.items()}
    for p in CLEAN_POLLUTANTS:
        stack = []
        for st in ds.stations:
            dates, v = prep.daily[st.slug][p]
            stack.append(cycles.seasonal_profile(dates, v))
        arr = np.array(stack)
        out["seasonal"][p] = [float(np.nanmean(arr[:, k])) if np.isfinite(arr[:, k]).any() else None for k in range(12)]
    return out, arrays


def weather_models(ds, prep, pollutants=("NO2", "PM10")):
    """Daily log-concentration models of weather, calendar and a time trend, scored by leave-one-year-out cross-validation."""
    rows = {}
    arrays = {}
    for st in _prepared(ds, prep):
        for p in pollutants:
            dates, v = prep.daily[st.slug][p]
            y = np.log(np.where(v > 0, v, np.nan))
            extra = weather.met_predictors(dates, prep.wdates, prep.wv)
            year = dates.astype("datetime64[Y]").astype(int) + 1970
            try:
                f, X, names = _model_fit(dates, y, extra, ds.holidays)
            except ValueError:
                continue
            pred_cv = np.full(len(y), np.nan)
            for yr in np.unique(year):
                test = year == yr
                if (np.isfinite(y[test]).sum() < 150):
                    continue
                try:
                    fy, Xy, _ = _model_fit(dates, y, extra, ds.holidays, train=~test, lags=0)
                except ValueError:
                    continue
                pred_cv[test] = (Xy @ fy["beta"])[test]
            ok = np.isfinite(y) & np.isfinite(pred_cv)
            if ok.sum() > 200:
                sse = float(np.sum((y[ok] - pred_cv[ok]) ** 2))
                sst = float(np.sum((y[ok] - y[ok].mean()) ** 2))
                r2_cv, r_cv = 1 - sse / sst, float(np.corrcoef(y[ok], pred_cv[ok])[0, 1])
            else:
                r2_cv = r_cv = float("nan")
            coef = {k: (float(f["beta"][names.index(k)]), float(f["se"][names.index(k)])) for k in ("log_wind", "log_blh", "temp", "rain", "years")}
            rows.setdefault(st.slug, {})[p] = {"r2": f["r2"], "r2_cv": r2_cv, "r_cv": r_cv, "n": f["n"], "n_cv": int(ok.sum()), "coef": coef}
            arrays[(st.slug, p)] = (dates, y, pred_cv)
    return rows, arrays


def trends(ds, prep, pollutants=("NO2", "PM10")):
    """Seasonal Mann-Kendall and Theil-Sen trends of monthly means, raw and after removing weather and calendar effects."""
    out = {}
    for st in _prepared(ds, prep):
        for p in pollutants:
            dates, v = prep.daily[st.slug][p]
            months, mm = trend.monthly_means(dates, v)
            entry = {}
            try:
                entry["raw"] = trend.trend_test(months, mm, reps=200)
            except ValueError:
                continue
            y = np.log(np.where(v > 0, v, np.nan))
            extra = weather.met_predictors(dates, prep.wdates, prep.wv)
            try:
                f, X, names = _model_fit(dates, y, extra, ds.holidays, trend_term=False)
                resid = np.full(len(y), np.nan)
                ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
                resid[ok] = (y - X @ f["beta"])[ok]
                level = np.nanmean(v)
                norm_daily = level * np.exp(resid)
                m2, mm2 = trend.monthly_means(dates, norm_daily)
                entry["normalised"] = trend.trend_test(m2, mm2, reps=200)
            except ValueError:
                pass
            out.setdefault(st.slug, {})[p] = entry
    return out


def traffic_summary(ds):
    t = ds.traffic
    prof = traffic.hourly_profiles(t, "traffic", ds.holidays)
    dates, v = daily(t, "traffic", 18)
    wdates, wv = weather.daily_weather(ds.weather)
    met = weather.met_predictors(dates, wdates, wv)
    rain_mm = np.full(len(dates), np.nan)
    pos = {int(d): i for i, d in enumerate(wdates.astype(int))}
    for i, d in enumerate(dates.astype(int)):
        if int(d) in pos:
            rain_mm[i] = wv["precipitation"][pos[int(d)]]
    hol_effect, hol_items = cycles.holiday_effect(dates, v, ds.holidays)
    rain = traffic.rain_effect(dates, v, rain_mm, ds.holidays)
    quiet = traffic.quiet_days(dates, v, ds.holidays)
    flagged = {str(x) for a, b, _ in quiet["events"] for x in np.arange(a, b + 1)}
    known_hits = [d for d in KNOWN_EXTENDED if d in flagged]
    # agreement of the hourly-derived daily mean with the independently published daily CSV
    cd = {str(d): a for d, a in zip(ds.traffic_daily["date"], ds.traffic_daily["average"])}
    pairs = np.array([(vv, cd[str(d)]) for d, vv in zip(dates, v) if str(d) in cd and np.isfinite(vv)])
    agree = {"days": int(len(pairs)), "r": float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]), "mean_hourly": float(pairs[:, 0].mean()), "mean_csv": float(pairs[:, 1].mean())}
    td = ds.traffic_daily
    years = td["date"].astype("datetime64[Y]").astype(int) + 1970
    annual = {int(y): float(np.mean(td["average"][years == y])) for y in np.unique(years)}
    del met
    return {"profiles": {k: {"peak_hour": int(np.nanargmax(m)), "peak": float(np.nanmax(m)), "night_min": float(np.nanmin(m))} for k, (m, lo, hi) in prof.items()},
            "holiday": hol_effect, "rain": rain, "quiet": [{"start": str(a), "end": str(b), "ratio": r} for a, b, r in quiet["events"]], "robust_sd": quiet["sd"],
            "known_extended": {"expected": list(KNOWN_EXTENDED), "detected": known_hits}, "source_agreement": agree, "annual_index": annual}, \
        {"profiles": prof, "dates": dates, "daily": v, "quiet_z": quiet["z"], "holiday_items": hol_items}


def lockdown_effects(ds, prep, pollutants=("NO2", "PM10")):
    """Weather-normalised change of pollution in the two lockdown windows against placebo windows, and the traffic change in the same windows.

    The model (calendar, weather, linear time trend) is fitted on all days outside the windows (plus a week on each side); the mean residual inside a window is
    ranked among the mean residuals of all other windows of equal length."""
    td = ds.traffic_daily
    traffic_change = {}
    for name, (a, b) in LOCKDOWNS.items():
        m = (td["date"] >= np.datetime64(a)) & (td["date"] <= np.datetime64(b))
        before = (td["date"] >= np.datetime64(a) - 28) & (td["date"] < np.datetime64(a))
        traffic_change[name] = {"index": float(td["average"][m].mean()), "before": float(td["average"][before].mean()),
                                "change": float(td["average"][m].mean() / td["average"][before].mean() - 1.0)}
    rows = {}
    for st in _prepared(ds, prep):
        for p in pollutants:
            dates, v = prep.daily[st.slug][p]
            y = np.log(np.where(v > 0, v, np.nan))
            extra = weather.met_predictors(dates, prep.wdates, prep.wv)
            outside = np.ones(len(dates), bool)
            for a, b in LOCKDOWNS.values():
                outside &= ~((dates >= np.datetime64(a, "D") - 7) & (dates <= np.datetime64(b, "D") + 7))
            try:
                f, X, names = _model_fit(dates, y, extra, ds.holidays, train=outside)
            except ValueError:
                continue
            resid = y - X @ f["beta"]
            for name, (a, b) in LOCKDOWNS.items():
                try:
                    w = nz.window_effect(resid, dates, a, b)
                except ValueError:
                    continue
                rows.setdefault(st.slug, {}).setdefault(p, {})[name] = {"change": float(np.exp(w["mean"]) - 1.0), "z": w["z"], "rank": w["rank"], "windows": w["windows"], "p": w["p"],
                                                                         "valid_days": w["valid_days"]}
    pooled = {}
    for p in pollutants:
        for name in LOCKDOWNS:
            ch = [w[name]["change"] for st in rows.values() for pp, w in st.items() if pp == p and name in w]
            if len(ch) >= 5:
                neg = int(sum(c < 0 for c in ch))
                pooled.setdefault(p, {})[name] = {"stations": len(ch), "median_change": float(np.median(ch)), "decreases": neg,
                                                  "sign_p": special.binom_two_sided_p(neg, len(ch), 0.5)}
    return {"traffic": traffic_change, "stations": rows, "pooled": pooled}


def episodes_summary(ds, prep, cams_daily):
    dates = prep.daily[ds.stations[0].slug]["PM10"][0]
    out = {}
    stack = np.array([prep.daily[s.slug]["PM10"][1] for s in ds.stations])
    med, cnt = episodes.regional_median(stack, min_stations=5)
    eps = episodes.find_episodes(dates, med, 50.0, min_days=1, max_gap=1)
    out["episodes"] = [{"start": str(e["start"]), "end": str(e["end"]), "days": e["days"], "peak": e["peak"], "mean": e["mean"]} for e in sorted(eps, key=lambda e: -e["peak"])[:15]]
    out["n_episodes"] = len(eps)
    out["episode_days"] = int(sum(e["days"] for e in eps))
    out["pc1_pm10"], out["pc1_days_pm10"] = episodes.first_component_share(stack, min_days=150)
    no2 = np.array([prep.daily[s.slug]["NO2"][1] for s in ds.stations if s.slug not in ("maslak",)])
    try:
        out["pc1_no2"], out["pc1_days_no2"] = episodes.first_component_share(no2, min_days=150)
    except ValueError:
        out["pc1_no2"], out["pc1_days_no2"] = None, 0
    cd_dates, cd = cams_daily["dates"], cams_daily["dust_max"]
    pos = {int(d): i for i, d in enumerate(cd_dates.astype(int))}
    idx = np.array([pos.get(int(d), -1) for d in dates.astype(int)])
    dust = np.where(idx >= 0, cd[np.maximum(idx, 0)], np.nan)
    ok = np.isfinite(med) & np.isfinite(dust)
    in_ep = np.zeros(len(dates), bool)
    for e in eps:
        in_ep |= (dates >= e["start"]) & (dates <= e["end"])
    thr = float(np.nanquantile(dust[ok], 0.95))
    out["dust_threshold"] = thr
    out["dust_overlap"] = episodes.overlap_table(in_ep[ok], (dust[ok] >= thr))
    ep_days_wind = np.array([cams_daily["wind_dir"][pos[int(d)]] if int(d) in pos else np.nan for d in dates.astype(int)])
    sel = in_ep & np.isfinite(ep_days_wind)
    if sel.sum() >= 5:
        mean_dir, r, p, z = episodes.rayleigh(ep_days_wind[sel])
        out["wind"] = {"mean_direction": mean_dir, "r": r, "p": p, "days": int(sel.sum()), "sector": weather.SECTOR_NAMES[int(weather.wind_sector(np.array([mean_dir]))[0])]}
    return out, {"dates": dates, "median": med, "dust": dust, "in_episode": in_ep}


def cams_daily(ds):
    c = ds.cams
    out = {}
    for name in ("pm10", "pm2_5", "nitrogen_dioxide", "ozone", "sulphur_dioxide", "carbon_monoxide"):
        dates, out[name] = daily(c, name, 18)
    dates, out["dust_max"] = daily(c, "dust", 18, how="max")
    wdates, wv = weather.daily_weather(ds.weather)
    wd_pos = {int(d): i for i, d in enumerate(wdates.astype(int))}
    out["wind_dir"] = np.array([wv["wind_direction"][wd_pos[int(d)]] if int(d) in wd_pos else np.nan for d in dates.astype(int)])
    out["dates"] = dates
    return out


def model_comparison(ds, prep, cams):
    pairs = (("PM10", "pm10"), ("NO2", "nitrogen_dioxide"), ("O3", "ozone"))
    out = {}
    cpos = {int(d): i for i, d in enumerate(cams["dates"].astype(int))}
    for st in _prepared(ds, prep):
        for p, mname in pairs:
            dates, v = prep.daily[st.slug][p]
            idx = np.array([cpos.get(int(d), -1) for d in dates.astype(int)])
            mod = np.where(idx >= 0, cams[mname][np.maximum(idx, 0)], np.nan)
            try:
                out.setdefault(st.slug, {})[p] = model.compare(v, mod)
            except ValueError:
                continue
    return out


def recent_summary(ds, days=30):
    """The last ``days`` days: daily AQI per station (from the published concentrations) and the traffic index."""
    rows = []
    for st in ds.stations:
        h = ds.air[st.slug]
        idx, dom, _ = aqi.aqi_series({p: h[p] for p in aqi.POLLUTANTS})
        dd = idx[-24 * days:]
        valid = dd[np.isfinite(dd)]
        last = float(valid[-1]) if len(valid) else None
        rows.append({"slug": st.slug, "name": st.name, "lat": st.lat, "lon": st.lon, "last_aqi": last, "last_time": str(h.times()[len(idx) - 1 - int(np.argmax(np.isfinite(dd[::-1])))]) if len(valid) else None,
                     "mean_aqi_30d": float(valid.mean()) if len(valid) else None, "max_aqi_30d": float(valid.max()) if len(valid) else None})
    return rows


def analyze(ds: Dataset) -> Results:
    prep = prepare(ds)
    summary = {"meta": ds.meta, "stations": station_overview(ds, prep)}
    arrays = {"prep": prep}
    summary["aqi_validation"] = aqi_validation(ds)
    summary["quality"] = quality_summary(ds, prep)
    summary["cycles"], arrays["profiles"] = cycles_summary(ds, prep)
    summary["weather_models"], arrays["weather_models"] = weather_models(ds, prep)
    summary["trends"] = trends(ds, prep)
    summary["traffic"], arrays["traffic"] = traffic_summary(ds)
    summary["lockdown"] = lockdown_effects(ds, prep)
    cams = cams_daily(ds)
    summary["episodes"], arrays["episodes"] = episodes_summary(ds, prep, cams)
    summary["model"] = model_comparison(ds, prep, cams)
    summary["recent"] = recent_summary(ds)
    arrays["cams"] = cams
    return Results(summary, arrays)

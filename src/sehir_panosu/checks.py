"""Validation: every estimator is confronted with a planted truth (synthetic data with a known answer) or with an independent source (the municipality's
own AQI, a second traffic feed, an atmospheric model, the government's announced holidays). ``run_all`` returns (name, passed, detail) triples; the thresholds follow
from the sampling distribution of the quantity, not from tuning to pass.
"""
from __future__ import annotations

import math

import numpy as np

from . import analysis, aqi, cycles, dataset, episodes, normalize as nz, quality, special, synth, traffic, trend
from .timeseries import daily


def _binom_ok(k, n, p, alpha=0.001):
    """True unless k successes in n trials are a (two-sided, exact) 1-in-1000 surprise under success probability p."""
    return special.binom_two_sided_p(k, n, p) > alpha


def check_special_functions():
    worst = 0.0
    worst = max(worst, abs(special.t_ppf(0.975, 10) - 2.228139), abs(special.t_ppf(0.975, 100) - 1.983972), abs(special.t_cdf(2.0, 1) - 0.852416))
    for x in (0.5, 2.0, 7.3):
        worst = max(worst, abs(special.chi2_sf(x, 2) - math.exp(-x / 2)))
    worst = max(worst, abs(special.norm_ppf(0.975) - 1.959964), abs(episodes.fisher_exact_two_sided(3, 1, 1, 3) - 0.485714))
    worst = max(worst, abs(special.binom_two_sided_p(3, 10, 0.5) - 0.34375))
    return worst < 1e-5, f"largest deviation from closed forms and tables: {worst:.1e}"


def check_aqi_against_published(ds, summary):
    v = summary["aqi_validation"]
    per = v["per_pollutant"]
    worst = min(per[p]["within1"] for p in aqi.POLLUTANTS)
    ok = v["overall"]["within1"] > 0.995 and worst > 0.995 and v["overall"]["n"] > 800_000
    return ok, (f"{v['overall']['n']:,} station-hours: the reconstructed overall AQI equals the published one after rounding in {100 * v['overall']['exact']:.1f} % and is within ±1 in "
                f"{100 * v['overall']['within1']:.2f} %; by pollutant at least {100 * worst:.2f} % within ±1")


def check_aqi_held_out(summary):
    h = summary["aqi_validation"]["held_out"]
    worst = min(e["test"]["within1"] for e in h.values())
    gap = max(abs(a - b) / b for e in h.values() for a, b in zip(e["knots"], e["table"]))
    return worst > 0.99 and gap < 0.01, (f"breakpoints fitted on stations {', '.join(summary['aqi_validation']['train_stations'][:3])}… reproduce the other stations within ±1 in at least "
                                          f"{100 * worst:.2f} % of hours; the refitted knots differ from the package's table by at most {100 * gap:.2f} %")


def check_knot_recovery(seed=1):
    rng = np.random.default_rng(seed)
    true = np.array([54.0, 154.0, 254.0])
    x = rng.uniform(0, 300, 20000)
    y = np.rint(np.interp(x, [0, *true], aqi.LEVELS[:4], right=150.0) + rng.normal(0, 0.3, len(x)))
    fit, rmse = aqi.fit_knots(x, y, [60.0, 140.0, 270.0])
    err = float(np.max(np.abs(fit - true)))
    return err < 1.0, f"planted knots {true.tolist()} recovered as {np.round(fit, 1).tolist()} (largest error {err:.2f} µg/m³, rmse {rmse:.2f})"


def check_quality_rules(reps=8, seed=2):
    rng = np.random.default_rng(seed)
    st_tp = st_n = st_fp = sp_tp = sp_n = sp_fp = 0
    for _ in range(reps):
        h = synth.make_series(730, rng, sigma=0.35, phi=0.9)
        f, stuck, spike = synth.plant_faults(np.round(h["x"], 1), rng)
        flags, _ = quality.flag_series("NO2", f)
        st_tp += int((flags["stuck"] & stuck).sum())
        st_n += int(stuck.sum())
        st_fp += int((flags["stuck"] & ~stuck).sum())
        sp_tp += int((flags["spike"] & spike).sum())
        sp_n += int(spike.sum())
        sp_fp += int((flags["spike"] & ~spike).sum())
    ok = st_tp / st_n > 0.98 and st_fp <= 10 and sp_tp / sp_n > 0.9 and sp_fp <= 2
    return ok, f"planted faults in {reps} two-year series: stuck hours found {st_tp}/{st_n} (false {st_fp}); spikes found {sp_tp}/{sp_n} (false {sp_fp})"


def check_weekend_effect(reps=150, seed=3):
    rng = np.random.default_rng(seed)
    out = {}
    for planted in (-0.2, 0.0):
        cover = rej = 0
        effs = []
        for _ in range(reps):
            h = synth.make_series(730, rng, weekend=planted, sigma=0.35, phi=0.9)
            d, v = daily(h, "x")
            e = cycles.weekend_effect(d, v)
            effs.append(e["effect"])
            cover += e["low"] <= math.exp(planted) <= e["high"]
            rej += e["p"] < 0.05
        out[planted] = (np.mean(effs), cover, rej)
    ok = _binom_ok(out[-0.2][1], reps, 0.95) and _binom_ok(out[0.0][1], reps, 0.95) and _binom_ok(out[0.0][2], reps, 0.05) and out[-0.2][2] == reps and abs(out[-0.2][0] + 0.2) < 0.03
    return ok, (f"planted weekend effect −0.20: estimated {out[-0.2][0]:.3f}, 95 % interval covers {out[-0.2][1]}/{reps}; no effect: rejects {out[0.0][2]}/{reps} at 5 % "
                f"(estimate {out[0.0][0]:+.3f}: the log of a mean over two days is biased low against five)")


def check_trend_test(reps=60, seed=4):
    rng = np.random.default_rng(seed)
    rej_null = rej_pow = 0
    sl = []
    for _ in range(reps):
        for planted in (0.0, -0.08):
            h = synth.make_series(365 * 7, rng, weekend=0, sigma=0.3, phi=0.85, trend_per_year=planted, seasonal=0.3)
            d, v = daily(h, "x")
            m, mm = trend.monthly_means(d, v)
            t = trend.trend_test(m, mm, reps=60, seed=1)
            if planted == 0.0:
                rej_null += t["p"] < 0.05
            else:
                rej_pow += t["p"] < 0.05
                sl.append(t["relative"])
    ok = rej_null <= 0.12 * reps and rej_pow == reps and abs(np.mean(sl) - (math.exp(-0.08) - 1)) < 0.015
    return ok, (f"seasonal Mann–Kendall on seven synthetic years: no trend rejected in {rej_null}/{reps} runs at 5 %; planted −8 %/yr detected {rej_pow}/{reps}, "
                f"Theil–Sen slope {100 * np.mean(sl):.1f} %/yr (truth {100 * (math.exp(-0.08) - 1):.1f})")


def check_regression(reps=120, seed=5):
    rng = np.random.default_rng(seed)
    cover = 0
    bs = []
    for _ in range(reps):
        s = synth.traffic_no2(1500, rng, elasticity=0.5, wind_beta=-0.4)
        X, names = nz.build_design(s["dates"], extra={"log_traffic": np.log(s["traffic"]), "log_wind": np.log(s["wind"])})
        f = nz.ols(X, np.log(s["no2"]), lags=14)
        i = names.index("log_traffic")
        bs.append(f["beta"][i])
        cover += abs(f["beta"][i] - 0.5) <= 1.96 * f["se"][i]
    ranks = []
    for _ in range(20):
        s = synth.traffic_no2(1500, rng, elasticity=0.5, drop_window=(900, 919), drop_depth=0.5)
        X, names = nz.build_design(s["dates"], extra={"log_wind": np.log(s["wind"])})
        f, use = nz.fit_complete_rows(X, np.log(s["no2"]), lags=14)
        res = np.full(len(s["dates"]), np.nan)
        res[use] = f["resid"]
        w = nz.window_effect(res, s["dates"], str(s["dates"][900]), str(s["dates"][918]))
        ranks.append(w["p"])
    detected = int(np.sum(np.array(ranks) < 0.05))
    ok = abs(np.mean(bs) - 0.5) < 0.02 and cover >= 0.88 * reps and detected >= 18
    return ok, (f"planted traffic elasticity 0.5: estimated {np.mean(bs):.3f} (sd {np.std(bs):.3f}), Newey–West 95 % interval covers {cover}/{reps}; "
                f"a planted 50 % traffic drop is outside the placebo windows in {detected}/20 runs")


def check_quiet_days(seed=6):
    rng = np.random.default_rng(seed)
    n = 1500
    dates = np.datetime64("2022-01-01") + np.arange(n)
    wd = (dates.astype(int) + 3) % 7
    month = dates.astype("datetime64[M]").astype(int) % 12
    base = 30.0 * np.where(wd >= 5, 0.75, 1.0) * (1 + 0.1 * np.cos(2 * np.pi * month / 12))
    v = base * np.exp(rng.normal(0, 0.12, n))
    planted = rng.choice(np.arange(100, n - 100, 7), 8, replace=False)
    v[planted] *= 0.45
    q = traffic.quiet_days(dates, v, [])
    flagged_all = set()
    for a, b, _ in q["events"]:
        flagged_all |= set(range(int(a.astype(int)), int(b.astype(int)) + 1))
    truth = {int(dates[i].astype(int)) for i in planted}
    hit = len(truth & flagged_all)
    false = len(flagged_all - truth)
    return hit >= 7 and false <= 3, f"8 planted quiet days (45 % of normal): found {hit}, false flags {false} among {n} days"


def check_circular_test(seed=7):
    rng = np.random.default_rng(seed)
    ps = [episodes.rayleigh(rng.uniform(0, 360, 60))[2] for _ in range(300)]
    ks = special.ks_uniform(np.array(ps))
    conc = episodes.rayleigh(rng.normal(150, 40, 60) % 360)
    return ks[1] > 0.001 and conc[2] < 1e-6 and abs(conc[0] - 150) < 15, f"Rayleigh test: p-values under uniform directions uniform (KS p = {ks[1]:.2f}); directions around 150° give p = {conc[2]:.1g} and mean {conc[0]:.0f}°"


def check_snapshot(ds):
    meta = ds.meta
    problems = []
    negatives = 0
    for st in ds.stations:
        h = ds.air[st.slug]
        if h.n != meta["air"]["hours"]:
            problems.append(f"{st.slug}: {h.n} hours")
        for p in ("NO2", "PM10"):
            x = h[p][np.isfinite(h[p])]
            negatives += int((x < 0).sum())
            if len(x) and x.max() > 5000:
                problems.append(f"{st.slug} {p}: implausible maximum {x.max()}")
        if not (40.8 < st.lat < 41.3 and 28.5 < st.lon < 29.4):
            problems.append(f"{st.slug}: position outside Istanbul")
    if ds.traffic.n != meta["traffic_hourly"]["hours"] or len(ds.traffic_daily["date"]) != meta["traffic_daily"]["days"]:
        problems.append("traffic sizes differ from the metadata")
    t = ds.traffic["traffic"][np.isfinite(ds.traffic["traffic"])]
    if t.min() < 0 or t.max() > 100:
        problems.append("traffic index outside 0-100")
    if negatives > 100:
        problems.append(f"{negatives} negative concentrations")
    return not problems, ("; ".join(problems) if problems else
                          f"{len(ds.stations)} stations × {meta['air']['hours']:,} hours, traffic {ds.traffic.n:,} hours and {len(ds.traffic_daily['date']):,} days: sizes match the metadata, "
                          f"positions plausible, {negatives} negative NO₂/PM10 readings in {len(ds.stations) * meta['air']['hours']:,} station-hours (the quality rules drop them)")


def check_traffic_sources(summary):
    a = summary["traffic"]["source_agreement"]
    return a["r"] > 0.95 and abs(a["mean_hourly"] - a["mean_csv"]) < 2.0, (f"hourly feed (daily means) against the published daily file: r = {a['r']:.3f} over {a['days']} days, "
                                                                          f"means {a['mean_hourly']:.1f} and {a['mean_csv']:.1f}")


def check_announced_holidays(summary):
    k = summary["traffic"]["known_extended"]
    return len(k["detected"]) == len(k["expected"]), f"announced extra holiday weekdays of 2024 flagged as unusually quiet: {len(k['detected'])} of {len(k['expected'])} ({', '.join(k['detected'])})"


def check_traffic_effects(summary):
    h, r = summary["traffic"]["holiday"], summary["traffic"]["rain"]
    ok = h["high"] < 0.8 and h["p"] < 0.001 and r["low"] > 1.0 and r["p"] < 0.001
    return ok, (f"weekday public holidays {100 * (h['ratio'] - 1):+.0f} % [{100 * (h['low'] - 1):+.0f}, {100 * (h['high'] - 1):+.0f}] over {h['events']} holidays; "
                f"rainy days {100 * (r['ratio'] - 1):+.1f} % [{100 * (r['low'] - 1):+.1f}, {100 * (r['high'] - 1):+.1f}]")


def check_model(summary):
    m = summary["model"]
    rs = [v[p]["r"] for v in m.values() for p in ("NO2", "PM10") if p in v]
    neg = sum(1 for v in m.values() if "NO2" in v and v["NO2"]["nmb"] < 0)
    n = sum(1 for v in m.values() if "NO2" in v)
    o3 = [v["O3"]["nmb"] for v in m.values() if "O3" in v]
    over = sum(1 for b in o3 if b > 0)
    ok = min(rs) > 0.3 and neg >= 0.75 * n and over >= 0.8 * len(o3)
    return ok, (f"CAMS against stations (daily means): NO₂ and PM10 correlations {min(rs):.2f}–{max(rs):.2f}; the model underestimates NO₂ at {neg} of {n} stations (a 10 km grid cannot resolve "
                f"street canyons) and overestimates O₃ at {over} of {len(o3)} (urban NO removes ozone at the stations)")


def check_episodes(summary):
    e = summary["episodes"]
    ov = e["dust_overlap"]
    return ov["p"] < 1e-6 and ov["odds_ratio"] > 3 and e["pc1_pm10"] > 0.5, (f"CAMS dust (independent of the stations) and station PM10 episodes: odds ratio {ov['odds_ratio']:.1f}, Fisher p {ov['p']:.1g}; "
                                                                               f"first principal component of PM10 {100 * e['pc1_pm10']:.0f} % of the variance")


def check_weather_models(summary):
    wm = summary["weather_models"]
    r2 = {p: np.nanmedian([e[p]["r2_cv"] for e in wm.values() if p in e]) for p in ("NO2", "PM10")}
    wind = [e["NO2"]["coef"]["log_wind"][0] for e in wm.values() if "NO2" in e]
    return r2["NO2"] > 0.2 and r2["PM10"] > 0.3 and all(w < 0 for w in wind), (f"leave-one-year-out R² of the daily weather models: median {r2['NO2']:.2f} (NO₂), {r2['PM10']:.2f} (PM10); "
                                                                                f"wind lowers NO₂ at all {len(wind)} stations (elasticity {min(wind):.2f} … {max(wind):.2f})")


def check_lockdown(summary):
    ld = summary["lockdown"]
    tr = {k: v["change"] for k, v in ld["traffic"].items()}
    hits = [(s, v["NO2"]["2021 full lockdown"]) for s, v in ld["stations"].items() if "NO2" in v and "2021 full lockdown" in v["NO2"] and v["NO2"]["2021 full lockdown"]["p"] < 0.05 and
            v["NO2"]["2021 full lockdown"]["change"] < 0]
    return all(c < -0.4 for c in tr.values()) and len(hits) >= 2, (f"traffic index in the lockdown windows: {', '.join(f'{k} {100 * c:+.0f} %' for k, c in tr.items())}; weather-normalised NO₂ beyond the 95 % of "
                                                                    f"placebo windows (2021) at {len(hits)} stations ({', '.join(s for s, _ in hits)})")


def run_all(quick=False, ds=None):
    ds = ds or dataset.load()
    summary = analysis.analyze(ds).summary
    checks = [("special functions against closed forms", check_special_functions),
              ("published AQI against the reconstruction", lambda: check_aqi_against_published(ds, summary)),
              ("AQI breakpoints fitted on half the stations", lambda: check_aqi_held_out(summary)),
              ("AQI knot recovery on planted breakpoints", check_knot_recovery),
              ("quality rules against planted faults", lambda: check_quality_rules(reps=4 if quick else 8)),
              ("weekend effect: recovery, coverage, false alarms", lambda: check_weekend_effect(reps=60 if quick else 150)),
              ("seasonal trend test: size, power, slope", lambda: check_trend_test(reps=25 if quick else 60)),
              ("traffic elasticity and placebo windows", lambda: check_regression(reps=50 if quick else 120)),
              ("quiet-day detector on planted days", check_quiet_days),
              ("Rayleigh and Fisher tests", check_circular_test),
              ("snapshot integrity", lambda: check_snapshot(ds)),
              ("two traffic sources against each other", lambda: check_traffic_sources(summary)),
              ("announced extended holidays", lambda: check_announced_holidays(summary)),
              ("holiday and rain effects on traffic", lambda: check_traffic_effects(summary)),
              ("weather models out of sample", lambda: check_weather_models(summary)),
              ("CAMS against stations", lambda: check_model(summary)),
              ("PM10 episodes against CAMS dust", lambda: check_episodes(summary)),
              ("lockdown windows", lambda: check_lockdown(summary))]
    out = []
    for name, fn in checks:
        ok, detail = fn()
        out.append((name, bool(ok), detail))
    return out


__all__ = ["run_all"]

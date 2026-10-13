"""Charts of the dashboard (matplotlib, Agg backend) returned as PNG bytes."""
from __future__ import annotations

import io
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import aqi  # noqa: E402

BLUE, ORANGE, GREEN, RED, SKY, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#6b7280"
INK, MUTED, GRID, LAND = "#1f2937", "#6b7280", "#e5e7eb", "#eef0f3"
POLLUTANT_LABEL = {"NO2": "NO₂", "PM10": "PM10", "O3": "O₃"}
POLLUTANT_COLOR = {"NO2": RED, "PM10": BLUE, "O3": GREEN}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#9ca3af", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def _once(seen, key, label):
    """``label`` the first time ``key`` is asked for, ``None`` afterwards (so each legend entry appears once even if the first station has no data)."""
    if key in seen:
        return None
    seen.add(key)
    return label


def to_png(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _coast(ax, coast, extent=(28.65, 29.32, 40.92, 41.2)):
    lon0, lon1, lat0, lat1 = extent
    ax.set_facecolor("#e6f0f7")
    for r in coast:
        if r["hole"]:
            continue
        xy = np.array(r["ring"])
        ax.fill(xy[:, 0], xy[:, 1], facecolor=LAND, edgecolor="#9ca3af", linewidth=0.7, zorder=0)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / math.cos(math.radians(0.5 * (lat0 + lat1))))
    ax.grid(False)
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")


def map_stations(coast, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, p, cmap in zip(axes, ("NO2", "PM10"), ("YlOrRd", "YlGnBu")):
        _coast(ax, coast)
        vals = np.array([r["mean"][p] if r["mean"][p] is not None else np.nan for r in rows])
        sc = ax.scatter([r["lon"] for r in rows], [r["lat"] for r in rows], c=vals, s=70 + 3.0 * np.nan_to_num(vals), cmap=cmap, edgecolors=INK, linewidths=0.6, zorder=3,
                        vmin=np.nanmin(vals) * 0.9, vmax=np.nanmax(vals) * 1.02)
        for r, v in zip(rows, vals):
            ax.annotate(r["name"], (r["lon"], r["lat"]), xytext=(5, 4), textcoords="offset points", fontsize=7.2, color=INK, zorder=4)
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label("mean, µg/m³ (2019–2026)")
        ax.set_title(f"Mean {POLLUTANT_LABEL[p]}", loc="left", fontsize=10)
    fig.tight_layout()
    return to_png(fig)


def chart_aqi(rows):
    order = sorted(rows, key=lambda r: -(r["aqi_mean"] or 0))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    left = np.zeros(len(order))
    for k, (_, name, _, color) in enumerate(aqi.CATEGORIES):
        share = np.array([r["aqi_shares"][k] if r["aqi_shares"] else 0.0 for r in order])
        ax.barh([r["name"] for r in order], 100 * share, left=left, color=color, label=name, height=0.7)
        left += 100 * share
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of hours (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7.5)
    return to_png(fig)


def chart_aqi_validation(val, sample):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1, 1.2]})
    mine, pub = sample
    a1.hexbin(pub, mine, gridsize=40, bins="log", cmap="Blues", mincnt=1, extent=(0, 200, 0, 200))
    a1.plot([0, 200], [0, 200], color=RED, lw=1)
    a1.set_xlabel("AQI published by İBB")
    a1.set_ylabel("AQI reconstructed here")
    a1.set_title("Hourly overall AQI, all stations", loc="left", fontsize=9.5)
    names = list(aqi.POLLUTANTS) + ["overall"]
    full = [val["per_pollutant"][p]["within1"] for p in aqi.POLLUTANTS] + [val["overall"]["within1"]]
    held = [val["held_out"][p]["test"]["within1"] for p in aqi.POLLUTANTS]
    x = np.arange(len(names))
    a2.bar(x - 0.2, [100 * v for v in full], 0.4, color=BLUE, label="all stations, table in the package")
    a2.bar(x[:-1] + 0.2, [100 * v for v in held], 0.4, color=ORANGE, label="held-out stations, breakpoints refitted on the others")
    a2.set_xticks(x)
    a2.set_xticklabels([POLLUTANT_LABEL.get(n, n) for n in names])
    a2.set_ylim(98.5, 100.05)
    a2.set_ylabel("hours within ±1 index point (%)")
    a2.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    a2.set_title("How often the reconstruction is within one point", loc="left", fontsize=9.5)
    fig.tight_layout()
    return to_png(fig)


def chart_quality(q, rows, slugs):
    years = q["years"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    names = {r["slug"]: r["name"] for r in rows}
    for ax, p in zip(axes, ("NO2", "PM10")):
        m = np.array([[q["completeness"][s][p].get(y, np.nan) for y in years] for s in slugs])
        ax.imshow(100 * m, cmap="Greens", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, fontsize=8)
        ax.set_yticks(range(len(slugs)))
        ax.set_yticklabels([names[s] for s in slugs], fontsize=8)
        ax.grid(False)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{100 * m[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if m[i, j] > 0.6 else INK)
        ax.set_title(f"{POLLUTANT_LABEL[p]}: share of hours with a value (%)", loc="left", fontsize=9.5)
    fig.tight_layout()
    return to_png(fig)


def chart_cycles(profiles, slugs, names):
    fig, axes = plt.subplots(len(slugs), 3, figsize=(10.8, 3.0 * len(slugs)), squeeze=False)
    hours = np.arange(24)
    for r, slug in enumerate(slugs):
        for c, p in enumerate(("NO2", "PM10", "O3")):
            ax = axes[r][c]
            prof = profiles.get((slug, p))
            if not prof:
                ax.axis("off")
                continue
            for label, color in (("weekday", POLLUTANT_COLOR[p]), ("weekend", GREY)):
                if label in prof:
                    mean, lo, hi = prof[label]
                    ax.plot(hours, mean, color=color, lw=2.0, label=label)
                    ax.fill_between(hours, lo, hi, color=color, alpha=0.18)
            ax.set_title(f"{names[slug]}: {POLLUTANT_LABEL[p]}", loc="left", fontsize=9)
            ax.set_xlabel("hour of day")
            ax.set_ylabel("µg/m³ (geometric mean)")
            ax.set_xticks(range(0, 24, 3))
            ax.legend(fontsize=7.5)
    fig.tight_layout()
    return to_png(fig)


def chart_weekend(weekend, rows):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.2), sharey=True)
    slugs = [r["slug"] for r in rows]
    names = [r["name"] for r in rows]
    for ax, p in zip(axes, ("NO2", "PM10", "O3")):
        for i, s in enumerate(slugs):
            e = weekend[s].get(p)
            if not e:
                continue
            ax.errorbar(e["ratio"], i, xerr=[[e["ratio"] - e["low"]], [e["high"] - e["ratio"]]], fmt="o", color=POLLUTANT_COLOR[p], ms=4, capsize=2, lw=1.3)
        ax.axvline(1.0, color=INK, lw=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("weekend / weekday (95 % interval)")
        ax.set_title(POLLUTANT_LABEL[p], loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return to_png(fig)


def chart_trends(trends, rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), sharey=True)
    names = {r["slug"]: r["name"] for r in rows}
    slugs = [r["slug"] for r in rows]
    seen = set()
    for ax, p in zip(axes, ("NO2", "PM10")):
        for i, s in enumerate(slugs):
            e = trends.get(s, {}).get(p)
            if not e:
                continue
            for off, key, color in ((-0.14, "raw", GREY), (0.14, "normalised", POLLUTANT_COLOR[p])):
                t = e.get(key)
                if t:
                    ax.errorbar(100 * t["relative"], i + off, xerr=[[100 * (t["relative"] - t["low"] / t["mean"])], [100 * (t["high"] / t["mean"] - t["relative"])]], fmt="o", ms=3.8,
                                color=color, capsize=1.8, lw=1.2, label=_once(seen, (id(ax), key), "raw monthly means" if key == "raw" else "weather-normalised"))
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(range(len(slugs)))
        ax.set_yticklabels([names[s] for s in slugs], fontsize=8)
        ax.set_xlabel("trend (% of the mean per year, 95 % interval)")
        ax.set_title(POLLUTANT_LABEL[p], loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].invert_yaxis()
    fig.tight_layout()
    return to_png(fig)


def chart_weather(wm, rows):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.2), sharey=True)
    names = {r["slug"]: r["name"] for r in rows}
    slugs = [r["slug"] for r in rows if r["slug"] in wm]
    seen = set()
    for ax, key, title in zip(axes, ("log_wind", "log_blh", "r2_cv"), ("log wind speed", "log boundary-layer height", "cross-validated R²")):
        for i, s in enumerate(slugs):
            for p, off in (("NO2", -0.15), ("PM10", 0.15)):
                e = wm[s].get(p)
                if not e:
                    continue
                if key == "r2_cv":
                    ax.plot(e["r2_cv"], i + off, "o", color=POLLUTANT_COLOR[p], ms=4, label=_once(seen, (id(ax), p), POLLUTANT_LABEL[p]))
                else:
                    b, se = e["coef"][key]
                    ax.errorbar(b, i + off, xerr=1.96 * se, fmt="o", color=POLLUTANT_COLOR[p], ms=4, capsize=2, lw=1.2, label=_once(seen, (id(ax), p), POLLUTANT_LABEL[p]))
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(range(len(slugs)))
        ax.set_yticklabels([names[s] for s in slugs], fontsize=8)
        ax.set_title(title if key == "r2_cv" else f"elasticity to {title}", loc="left", fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=7.5)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return to_png(fig)


def chart_traffic(arrays, summary):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1, 1.5]})
    hours = np.arange(24)
    colors = {"weekday": BLUE, "saturday": ORANGE, "sunday": GREEN, "holiday": RED}
    for k, (mean, lo, hi) in arrays["profiles"].items():
        a1.plot(hours, mean, color=colors[k], lw=2, label=k)
        a1.fill_between(hours, lo, hi, color=colors[k], alpha=0.15)
    a1.set_xlabel("hour of day")
    a1.set_ylabel("traffic index (0–100)")
    a1.set_xticks(range(0, 24, 3))
    a1.legend(fontsize=8)
    a1.set_title("Daily curves, Nov 2021 – Oct 2026", loc="left", fontsize=9.5)
    dates, v = arrays["dates"], arrays["daily"]
    a2.plot(dates, v, color=BLUE, lw=0.8)
    quiet = np.zeros(len(dates), bool)
    for e in summary["quiet"]:
        quiet |= (dates >= np.datetime64(e["start"], "D")) & (dates <= np.datetime64(e["end"], "D"))
    a2.scatter(dates[quiet], v[quiet], s=16, color=RED, zorder=3)
    for d in summary["known_extended"]["expected"]:
        a2.plot(np.datetime64(d), 3, marker="^", color=INK, ms=4, lw=0)
    a2.set_ylabel("daily mean traffic index")
    a2.set_title("Days quieter than weekday, month and holidays explain (red); ▲ = announced extended holiday", loc="left", fontsize=8.8)
    fig.tight_layout()
    return to_png(fig)


def chart_lockdown(ds, summary, rows):
    td = ds.traffic_daily
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    m = (td["date"] >= np.datetime64("2019-06-01")) & (td["date"] <= np.datetime64("2022-01-01"))
    a1.plot(td["date"][m], td["average"][m], color=BLUE, lw=0.8)
    from .analysis import LOCKDOWNS
    for name, (a, b) in LOCKDOWNS.items():
        a1.axvspan(np.datetime64(a), np.datetime64(b), color=RED, alpha=0.25, lw=0)
        a1.text(np.datetime64(a), 52, name, fontsize=7.5, rotation=90, va="top", ha="right", color=RED)
    a1.set_ylabel("daily mean traffic index")
    a1.set_title("Traffic collapses in both lockdown windows", loc="left", fontsize=9.5)
    st = summary["stations"]
    names = {r["slug"]: r["name"] for r in rows}
    slugs = [r["slug"] for r in rows]
    seen = set()
    for i, s in enumerate(slugs):
        for p, off, mk in (("NO2", -0.15, "o"), ("PM10", 0.15, "s")):
            e = st.get(s, {}).get(p, {}).get("2021 full lockdown")
            if e:
                a2.plot(100 * e["change"], i + off, mk, color=POLLUTANT_COLOR[p], ms=4.5, label=_once(seen, p, POLLUTANT_LABEL[p]), mfc=POLLUTANT_COLOR[p] if e["p"] < 0.05 else "white")
    a2.axvline(0, color=INK, lw=0.8)
    a2.set_yticks(range(len(slugs)))
    a2.set_yticklabels([names[s] for s in slugs], fontsize=8)
    a2.invert_yaxis()
    a2.set_xlabel("weather-normalised change, 29 Apr – 17 May 2021 (%)\nfilled = beyond 95 % of placebo windows")
    a2.grid(axis="y", visible=False)
    a2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return to_png(fig)


def chart_episodes(arr, summary):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [2.2, 1]})
    dates, med = arr["dates"], arr["median"]
    a1.plot(dates, med, color=BLUE, lw=0.7, label="regional daily PM10 (median of stations)")
    a1.axhline(50, color=RED, lw=1, ls="--", label="50 µg/m³ (EU daily limit)")
    a1.axhline(45, color=ORANGE, lw=1, ls=":", label="45 µg/m³ (WHO 2021)")
    a1.set_ylabel("µg/m³")
    a1.legend(fontsize=7.5, loc="upper left")
    a1b = a1.twinx()
    a1b.plot(dates, arr["dust"], color=ORANGE, lw=0.6, alpha=0.8)
    a1b.set_ylabel("CAMS dust, daily max (µg/m³)", color=ORANGE)
    a1b.grid(False)
    a1b.spines["right"].set_visible(True)
    a1.set_title(f"{summary['n_episodes']} regional PM10 episodes ({summary['episode_days']} days)", loc="left", fontsize=9.5)
    ov = summary["dust_overlap"]
    cells = np.array([[ov["both"], ov["episode_only"]], [ov["signal_only"], ov["neither"]]], float)
    a2.imshow(np.log10(cells + 1), cmap="Blues")
    for i in range(2):
        for j in range(2):
            a2.text(j, i, f"{int(cells[i, j])}", ha="center", va="center", fontsize=11, color="white" if np.log10(cells[i, j] + 1) > 1.8 else INK)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["episode", "no episode"])
    a2.set_yticks([0, 1])
    a2.set_yticklabels(["CAMS dust\nhigh (top 5 %)", "dust not high"])
    a2.grid(False)
    a2.set_title(f"Days: odds ratio {ov['odds_ratio']:.1f}, Fisher p {ov['p']:.1g}", loc="left", fontsize=9)
    fig.tight_layout()
    return to_png(fig)


def chart_model(model_rows, rows):
    names = {r["slug"]: r["name"] for r in rows}
    slugs = [r["slug"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=True)
    seen = set()
    for ax, key, title, xlim in zip(axes, ("nmb", "r"), ("normalised mean bias (model − station) / station", "correlation of daily means"), ((-0.8, 0.4), (0.2, 0.95))):
        for i, s in enumerate(slugs):
            for p, off in (("NO2", -0.15), ("PM10", 0.15)):
                e = model_rows.get(s, {}).get(p)
                if not e:
                    continue
                if key == "r":
                    ax.errorbar(e["r"], i + off, xerr=[[e["r"] - e["r_low"]], [e["r_high"] - e["r"]]], fmt="o", color=POLLUTANT_COLOR[p], ms=4, capsize=2, lw=1.2, label=_once(seen, (id(ax), p), POLLUTANT_LABEL[p]))
                else:
                    ax.plot(e["nmb"], i + off, "o", color=POLLUTANT_COLOR[p], ms=4.5, label=_once(seen, (id(ax), p), POLLUTANT_LABEL[p]))
        if key == "nmb":
            ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlim(*xlim)
        ax.set_yticks(range(len(slugs)))
        ax.set_yticklabels([names[s] for s in slugs], fontsize=8)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return to_png(fig)


def chart_recent(rows, coast):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    _coast(ax, coast)
    xs = [r["lon"] for r in rows]
    ys = [r["lat"] for r in rows]
    colors = [aqi.category(r["last_aqi"] if r["last_aqi"] is not None else float("nan"))[2] for r in rows]
    ax.scatter(xs, ys, c=colors, s=260, edgecolors=INK, linewidths=0.8, zorder=3)
    for r in rows:
        label = r["name"] + ("" if r["last_aqi"] is None else f"  {r['last_aqi']:.0f}")
        ax.annotate(label, (r["lon"], r["lat"]), xytext=(8, 5), textcoords="offset points", fontsize=7.8, zorder=4)
    ax.set_title("Latest AQI per station (colour = İBB category)", loc="left", fontsize=10)
    return to_png(fig)


def chart_traffic_now(panel):
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    t = panel["times"].astype("datetime64[h]")
    ax.fill_between(t, panel["q25"], panel["q75"], color=SKY, alpha=0.35, label="typical range for this weekday and hour (quartiles)")
    ax.plot(t, panel["median"], color=SKY, lw=1.5, label="typical median")
    ax.plot(t, panel["values"], color=BLUE, lw=2.0, label="traffic index")
    ax.set_ylabel("traffic index (0–100)")
    ax.legend(fontsize=7.5, loc="upper left", ncol=3)
    ax.set_title(f"Last {len(t)} hours; latest {panel['latest']:.0f} at {panel['latest_time'].replace('T', ' ')}:00", loc="left", fontsize=9.5)
    fig.autofmt_xdate()
    return to_png(fig)

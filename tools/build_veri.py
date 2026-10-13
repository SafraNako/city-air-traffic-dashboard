"""Build the data snapshot shipped in ``src/sehir_panosu/veri`` from the open-data services.

Usage: python tools/build_veri.py RAW_DIR [--start 2019-01-01] [--end YYYY-MM-DD] [--refetch] [--workers N]

Every raw service answer is cached in RAW_DIR (so the build can be repeated offline); missing pieces are fetched with the package's own fetchers
(``sehir_panosu.bundle``). If RAW_DIR holds ``ne_10m_land.geojson`` (Natural Earth, public domain) the coastline of the maps is rebuilt from it.
"""
import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT = os.path.join(os.path.dirname(__file__), "..", "src", "sehir_panosu", "veri")
BBOX = (40.6, 41.5, 28.2, 29.8)            # lat_min, lat_max, lon_min, lon_max of the coastline kept


def clip_ring(ring, lat0, lat1, lon0, lon1):
    """Sutherland-Hodgman clipping of a ring of (lon, lat) points against a rectangle."""
    def clip(points, inside, cross):
        out = []
        for i, cur in enumerate(points):
            prev = points[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out

    def x_at(p, q, x):
        t = (x - p[0]) / (q[0] - p[0])
        return (x, p[1] + t * (q[1] - p[1]))

    def y_at(p, q, y):
        t = (y - p[1]) / (q[1] - p[1])
        return (p[0] + t * (q[0] - p[0]), y)

    pts = [tuple(p) for p in ring[:-1]] if ring[0] == ring[-1] else [tuple(p) for p in ring]
    for inside, cross in ((lambda p: p[0] >= lon0, lambda p, q: x_at(p, q, lon0)), (lambda p: p[0] <= lon1, lambda p, q: x_at(p, q, lon1)),
                          (lambda p: p[1] >= lat0, lambda p, q: y_at(p, q, lat0)), (lambda p: p[1] <= lat1, lambda p, q: y_at(p, q, lat1))):
        if not pts:
            break
        pts = clip(pts, inside, cross)
    return pts


def build_coast(raw):
    data = json.load(open(os.path.join(raw, "ne_10m_land.geojson"), encoding="utf-8"))
    lat0, lat1, lon0, lon1 = BBOX
    rings = []
    for f in data["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            for k, ring in enumerate(poly):
                xs = [p[0] for p in ring]
                ys = [p[1] for p in ring]
                if max(xs) < lon0 or min(xs) > lon1 or max(ys) < lat0 or min(ys) > lat1:
                    continue
                clipped = clip_ring(ring, lat0, lat1, lon0, lon1)
                if len(clipped) >= 3:
                    rings.append({"hole": k > 0, "ring": [[round(x, 4), round(y, 4)] for x, y in clipped]})
    return rings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    from sehir_panosu import bundle
    coast = build_coast(a.raw) if os.path.exists(os.path.join(a.raw, "ne_10m_land.geojson")) else None
    meta = bundle.build(OUT, raw=a.raw, start=a.start, end=a.end, refetch=a.refetch, workers=a.workers)
    if coast is not None:
        with open(os.path.join(OUT, "kiyi.json"), "w", encoding="utf-8") as f:
            json.dump(coast, f, separators=(",", ":"))
    print(json.dumps(meta, indent=1))
    for name in sorted(os.listdir(OUT)):
        print(f"{os.path.getsize(os.path.join(OUT, name)) / 1024:9.0f} KB  {name}")


if __name__ == "__main__":
    main()

"""Write examples/pano.html and the PNG charts used by the README."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sehir_panosu import dashboard  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "examples")

dash = dashboard.build()
with open(os.path.join(OUT, "pano.html"), "w", encoding="utf-8") as f:
    f.write(dash.render())
for key, name in (("map", "station_map.png"), ("aqival", "aqi_reconstruction.png"), ("cycles", "daily_rhythm.png"), ("traffic", "traffic.png"), ("lockdown", "lockdown.png"),
                  ("episodes", "episodes.png"), ("model", "cams_vs_stations.png"), ("trends", "trends.png")):
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(dash.charts[key])
for name in sorted(os.listdir(OUT)):
    print(f"{os.path.getsize(os.path.join(OUT, name)) / 1024:8.0f} KB  {name}")

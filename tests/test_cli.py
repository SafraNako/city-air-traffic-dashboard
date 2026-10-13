import os

import pytest

from sehir_panosu import cli


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_ozet(capsys):
    code, out, _ = run(capsys, "ozet")
    assert code == 0 and "13 stations, 68,232 hourly values from 2019-01-01" in out and "111 public holidays" in out
    assert "Aksaray" in out and "Üsküdar 1" in out and out.count("\n") >= 16


def test_simdi_snapshot_and_live(capsys, upstream):
    code, out, err = run(capsys, "simdi")
    assert code == 0 and "category" in out and "Traffic index" in out and "Live" not in out and err == "" and upstream.count("/air/GetAQIByStationId") == 0
    code, out, err = run(capsys, "simdi", "--canli", "--gun", "2")
    assert code == 0 and "Live: 13 stations" in out and upstream.count("/air/GetAQIByStationId") == 13


def test_simdi_live_failure_falls_back(capsys, upstream, monkeypatch):
    from sehir_panosu import sources
    monkeypatch.setattr(sources, "RETRY_DELAY", 0.0)
    upstream.fail = True
    code, out, err = run(capsys, "simdi", "--canli")
    assert code == 0 and "live fetch failed" in err and "Traffic index" in out


def test_aqi_check_and_calculator(capsys):
    code, out, _ = run(capsys, "aqi")
    assert code == 0 and "within 1" in out and "overall" in out and "refitted" in out and "PM10" in out
    code, out, _ = run(capsys, "aqi", "--pm10", "120", "--no2", "150", "--o3", "90")
    assert code == 0 and "AQI 83 (PM10): Moderate / Orta" in out
    code, out, _ = run(capsys, "aqi", "--pm10", "30", "--no2", "20")
    assert "AQI 28 (PM10)" in out and "Good / İyi" in out
    code, out, _ = run(capsys, "aqi", "--no2", "300")
    assert "Unhealthy for sensitive groups" in out


def test_istasyon(capsys):
    code, out, _ = run(capsys, "istasyon", "--ad", "aksaray")
    assert code == 0 and "Aksaray (41.0144 N, 28.9540 E)" in out and "share of hours by AQI category" in out and "Hazardous" in out and "annual means" in out and "NO2:" in out
    code, out, _ = run(capsys, "istasyon", "--ad", "Ümraniye 1")
    assert code == 0 and "Ümraniye 1" in out
    code, _, err = run(capsys, "istasyon", "--ad", "nowhere")
    assert code == 2 and "unknown station" in err


def test_ritim(capsys):
    code, out, _ = run(capsys, "ritim", "--istasyon", "aksaray", "--kirletici", "NO2")
    assert code == 0 and "weekday" in out and "weekend" in out and "weekend / weekday = 0.9" in out and out.count("\n") > 26
    code, out, _ = run(capsys, "ritim", "--istasyon", "avcilar", "--kirletici", "O3")
    assert code == 0 and "weekend / weekday = 1." in out


def test_trend(capsys):
    code, out, _ = run(capsys, "trend", "--istasyon", "selimiye", "--kirletici", "NO2", "--tekrar", "100")
    assert code == 0 and "seasonal Mann-Kendall" in out and "per year" in out and "-1" in out
    code, _, err = run(capsys, "trend", "--istasyon", "x", "--kirletici", "NO2")
    assert code == 2 and "unknown station" in err


def test_trafik(capsys):
    code, out, _ = run(capsys, "trafik", "--adet", "5")
    assert code == 0 and "weekday public holidays: -41 %" in out and "rainy days" in out and "announced extra holiday weekdays of 2024 flagged: 4 of 4" in out and "unusually quiet" in out
    assert out.count(" of expected") == 5


def test_kilitlenme(capsys):
    code, out, _ = run(capsys, "kilitlenme")
    assert code == 0 and "2021 full lockdown" in out and "aksaray" in out and "sign test p" in out and "Across stations" in out


def test_olay_and_model(capsys):
    code, out, _ = run(capsys, "olay", "--adet", "3")
    assert code == 0 and "regional PM10 episodes" in out and "odds ratio" in out and "Rayleigh" in out and "2022-12-30" in out
    code, out, _ = run(capsys, "model")
    assert code == 0 and "CAMS mean" in out and out.count("Aksaray") == 3                                  # NO2, PM10 and O3 rows


def test_indir_writes_a_usable_bundle(capsys, upstream, tmp_path):
    out_dir, raw = str(tmp_path / "veri"), str(tmp_path / "raw")
    code, out, _ = run(capsys, "indir", "--cikti", out_dir, "--ham", raw, "--baslangic", "2026-10-10", "--bitis", "2026-10-12", "--paralel", "2")
    assert code == 0 and "Done: 13 stations, 72 hours" in out and "SEHIR_VERI_DIZINI" in out and os.path.exists(os.path.join(raw, "stations.json"))
    code, out, _ = run(capsys, "--veri-dizini", out_dir, "ozet")
    assert code == 0 and "13 stations, 72 hourly values from 2026-10-10" in out


def test_indir_failure_is_reported(capsys, upstream, tmp_path, monkeypatch):
    from sehir_panosu import sources
    monkeypatch.setattr(sources, "RETRY_DELAY", 0.0)
    upstream.fail = True
    code, _, err = run(capsys, "indir", "--cikti", str(tmp_path / "x"), "--baslangic", "2026-10-10", "--bitis", "2026-10-12")
    assert code == 2 and err.startswith("error:")


def test_pano_writes_the_dashboard(capsys, tmp_path):
    target = tmp_path / "pano.html"
    code, out, _ = run(capsys, "pano", "--cikti", str(target))
    assert code == 0 and "pano.html" in out and target.stat().st_size > 1_000_000
    assert "Istanbul air and traffic dashboard" in target.read_text(encoding="utf-8")


def test_parser_and_unknown_commands():
    p = cli.build_parser()
    a = p.parse_args(["sunucu", "--port", "0", "--canli", "--aralik", "60", "--adres", "0.0.0.0"])
    assert (a.port, a.canli, a.aralik, a.adres) == (0, True, 60, "0.0.0.0") and a.fn is cli.cmd_sunucu
    assert p.parse_args(["dogrulama", "--hizli"]).hizli is True and p.parse_args(["aqi", "--pm10", "5"]).pm10 == 5.0
    with pytest.raises(SystemExit):
        p.parse_args([])
    with pytest.raises(SystemExit):
        p.parse_args(["nonsense"])
    with pytest.raises(SystemExit):
        p.parse_args(["ritim", "--istasyon", "aksaray", "--kirletici", "CO"])

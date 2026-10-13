import numpy as np
import pytest
from conftest import fixture_bytes

from sehir_panosu import sources, weather
from sehir_panosu.timeseries import Hourly


@pytest.fixture(scope="module")
def sample():
    return sources.parse_open_meteo(fixture_bytes("weather_sample.json"))


def test_daily_weather_on_the_real_sample(sample):
    dates, wv = weather.daily_weather(sample)
    assert dates[0] == np.datetime64("2024-04-06") and len(dates) == 9
    assert wv["temperature_2m"][0] == pytest.approx(np.mean(sample["temperature_2m"][:24]))
    assert wv["precipitation"][0] == pytest.approx(np.sum(sample["precipitation"][:24]))             # mm per day
    assert wv["wind_speed_10m"][1] == pytest.approx(np.mean(sample["wind_speed_10m"][24:48]))
    assert np.all((wv["wind_direction"] >= 0) & (wv["wind_direction"] < 360))


def test_daily_wind_direction_is_the_vector_mean():
    n = 48
    h = Hourly("2024-01-01T00", n, {"wind_speed_10m": np.full(n, 10.0), "wind_direction_10m": np.r_[np.full(24, 90.0), np.full(24, 350.0)], "precipitation": np.zeros(n)})
    _, wv = weather.daily_weather(h)
    assert wv["wind_direction"][0] == pytest.approx(90.0) and wv["wind_direction"][1] == pytest.approx(350.0)
    mixed = Hourly("2024-01-01T00", 24, {"wind_speed_10m": np.full(24, 5.0), "wind_direction_10m": np.r_[np.full(12, 350.0), np.full(12, 10.0)], "precipitation": np.zeros(24)})
    north = weather.daily_weather(mixed)[1]["wind_direction"][0]
    assert north < 0.1 or north > 359.9                                                   # 350 and 10 degrees average to north, not to 180


def test_wind_sectors():
    d = np.array([0.0, 22.0, 23.0, 90.0, 180.0, 270.0, 337.0, 338.0, 359.0, np.nan])
    assert weather.wind_sector(d).tolist() == [0, 0, 1, 2, 4, 6, 7, 0, 0, -1]
    assert weather.SECTOR_NAMES[2] == "E" and len(weather.SECTOR_NAMES) == 8


def test_met_predictors_alignment_and_lags(sample):
    wdates, wv = weather.daily_weather(sample)
    dates = np.datetime64("2024-04-05") + np.arange(12)                       # one day before and two after the weather series
    m = weather.met_predictors(dates, wdates, wv)
    assert np.isnan(m["log_wind"][0]) and np.isnan(m["temp"][0]) and np.isfinite(m["temp"][1]) and np.isnan(m["temp"][-1])
    assert m["log_wind"][1] == pytest.approx(np.log(max(wv["wind_speed_10m"][0], 0.5)))
    assert m["log_wind_lag1"][2] == pytest.approx(m["log_wind"][1]) and np.isnan(m["log_wind_lag1"][0])
    assert set(m) == {"log_wind", "log_wind_lag1", "log_blh", "temp", "rain", "humidity", "log_rad", "dir_sin", "dir_cos"}
    rain = m["rain"][np.isfinite(m["rain"])]
    assert set(np.unique(rain)) <= {0.0, 1.0} and np.allclose(m["dir_sin"][1:10] ** 2 + m["dir_cos"][1:10] ** 2, 1.0)


def test_real_weather_is_sane(ds):
    wdates, wv = weather.daily_weather(ds.weather)
    assert 12 < np.nanmean(wv["temperature_2m"]) < 17 and 10 < np.nanmean(wv["wind_speed_10m"]) < 18 and 200 < np.nanmean(wv["boundary_layer_height"]) < 900
    assert 400 < np.nansum(wv["precipitation"]) / (len(wdates) / 365.25) < 1000                                # mm per year
    sector = weather.wind_sector(wv["wind_direction"])
    counts = np.bincount(sector[sector >= 0], minlength=8)
    assert counts.argmax() in (0, 1, 7, 5, 4, 6)                                                              # the dominant flow is northerly or south-westerly in Istanbul

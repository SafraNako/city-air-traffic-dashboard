import numpy as np
import pytest

from sehir_panosu import normalize as nz
from sehir_panosu import synth


def test_ols_matches_the_closed_form_and_reports_r2():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 2.0 + 3.0 * x + rng.normal(0, 0.5, 500)
    X = np.column_stack([np.ones(500), x])
    f = nz.ols(X, y)
    assert f["beta"] == pytest.approx([2.0, 3.0], abs=0.15) and f["n"] == 500
    ref = np.linalg.solve(X.T @ X, X.T @ y)
    assert f["beta"] == pytest.approx(ref, rel=1e-9)
    assert f["se"][1] == pytest.approx(0.5 / np.sqrt(500), rel=0.25) and 0.9 < f["r2"] < 1 and f["p"][1] < 1e-10
    assert f["fitted"] + f["resid"] == pytest.approx(y)


def test_newey_west_errors_are_wider_under_autocorrelation_and_similar_without():
    rng = np.random.default_rng(1)
    n = 3000
    x = synth.ar1(n, 0.9, 1.0, rng)
    y = 1.0 * x + synth.ar1(n, 0.9, 1.0, rng)
    X = np.column_stack([np.ones(n), x])
    plain, hac = nz.ols(X, y, 0), nz.ols(X, y, 30)
    assert hac["se"][1] > 1.5 * plain["se"][1]
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    Xw = np.column_stack([np.ones(n), z])
    assert nz.ols(Xw, w, 10)["se"][1] == pytest.approx(nz.ols(Xw, w, 0)["se"][1], rel=0.2)


def test_calendar_columns_and_design():
    dates = np.datetime64("2020-01-01") + np.arange(14)                   # Wednesday 1 January 2020
    cols, names = nz.calendar_columns(dates, ["2020-01-06"])
    assert names[:6] == [f"weekday{k}" for k in range(1, 7)] and names[-1] == "holiday" and len(cols) == 6 + 11 + 1
    wd1 = cols[names.index("weekday1")]                                    # Tuesdays
    assert wd1.sum() == 2 and wd1[6] == 1.0                                # 7 January is the 7th element (index 6)
    assert cols[names.index("holiday")].tolist().count(1.0) == 1 and cols[names.index("holiday")][5] == 1.0
    X, nm = nz.build_design(dates, [], {"extra": np.arange(14.0)})
    assert nm[0] == "const" and nm[-1] == "extra" and X.shape == (14, 1 + 6 + 11 + 1 + 1) and np.all(X[:, 0] == 1)
    X2, _ = nz.build_design(dates, [], intercept=False)
    assert X2.shape[1] == 6 + 11 + 1


def test_holidays_on_weekends_are_not_marked():
    dates = np.datetime64("2020-01-04") + np.arange(3)                     # Saturday .. Monday
    cols, names = nz.calendar_columns(dates, ["2020-01-04", "2020-01-06"])
    assert cols[names.index("holiday")].tolist() == [0.0, 0.0, 1.0]


def test_fit_complete_rows_skips_nan_and_respects_the_training_mask():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    y = 1.0 + 2.0 * x + rng.normal(0, 0.1, 400)
    y[::10] = np.nan
    X = np.column_stack([np.ones(400), x])
    X[5, 1] = np.nan
    f, use = nz.fit_complete_rows(X, y)
    assert use.sum() == 400 - 40 - 1                                      # every tenth y and one row of X are NaN
    assert f["beta"] == pytest.approx([1.0, 2.0], abs=0.05)
    train = np.arange(400) < 150
    f2, use2 = nz.fit_complete_rows(X, y, train=train)
    assert use2.sum() < use.sum() and not use2[200:].any()
    with pytest.raises(ValueError):
        nz.fit_complete_rows(X, y, train=np.arange(400) < 30)


def test_window_effect_detects_a_planted_dip_and_ranks_it():
    rng = np.random.default_rng(3)
    dates = np.datetime64("2020-01-01") + np.arange(1500)
    resid = synth.ar1(1500, 0.6, 0.1, rng)
    resid[600:620] -= 0.4
    w = nz.window_effect(resid, dates, str(dates[600]), str(dates[619]))
    assert w["mean"] == pytest.approx(-0.4, abs=0.12) and w["rank"] == 1 and w["z"] < -4 and w["p"] < 0.01 and w["valid_days"] == 20 and w["windows"] > 1300
    quiet = nz.window_effect(resid, dates, str(dates[1000]), str(dates[1019]))
    assert quiet["p"] > 0.05 and abs(quiet["z"]) < 3
    with pytest.raises(ValueError):
        nz.window_effect(resid, dates, str(dates[10]), str(dates[11]))


def test_window_effect_needs_valid_data():
    dates = np.datetime64("2020-01-01") + np.arange(300)
    resid = np.full(300, np.nan)
    resid[:100] = 0.0
    with pytest.raises(ValueError):
        nz.window_effect(resid, dates, str(dates[150]), str(dates[170]))


def test_lagged():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(nz.lagged(x, 1), [np.nan, 1, 2, 3], equal_nan=True) and np.array_equal(nz.lagged(x, 0), x)
    assert np.allclose(nz.lagged(x, -1), [2, 3, 4, np.nan], equal_nan=True)


def test_elasticity_and_wind_effect_are_recovered_with_honest_errors():
    rng = np.random.default_rng(4)
    reps, cover, bs = 60, 0, []
    for _ in range(reps):
        s = synth.traffic_no2(1500, rng, elasticity=0.5, wind_beta=-0.4)
        X, names = nz.build_design(s["dates"], extra={"log_traffic": np.log(s["traffic"]), "log_wind": np.log(s["wind"])})
        f = nz.ols(X, np.log(s["no2"]), lags=14)
        i = names.index("log_traffic")
        bs.append(f["beta"][i])
        cover += abs(f["beta"][i] - 0.5) <= 1.96 * f["se"][i]
    assert abs(np.mean(bs) - 0.5) < 0.03 and cover >= 0.8 * reps

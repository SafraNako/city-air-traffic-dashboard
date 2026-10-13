import numpy as np
import pytest

from sehir_panosu import model, synth


def test_perfect_and_biased_models():
    rng = np.random.default_rng(0)
    obs = np.exp(rng.normal(3.5, 0.5, 500))
    perfect = model.compare(obs, obs)
    assert perfect["bias"] == 0 and perfect["nmb"] == 0 and perfect["rmse"] == 0 and perfect["r"] == pytest.approx(1.0) and perfect["slope"] == pytest.approx(1.0)
    low = model.compare(obs, 0.6 * obs)
    assert low["nmb"] == pytest.approx(-0.4) and low["bias"] == pytest.approx(-0.4 * obs.mean()) and low["slope"] == pytest.approx(0.6) and low["r"] == pytest.approx(1.0)
    assert low["obs_mean"] == pytest.approx(obs.mean()) and low["mod_mean"] == pytest.approx(0.6 * obs.mean())


def test_nan_pairs_are_dropped_and_small_samples_refused():
    obs = np.arange(1.0, 101.0)
    mod = obs + np.random.default_rng(1).normal(0, 3, 100)
    obs2, mod2 = obs.copy(), mod.copy()
    obs2[:20] = np.nan
    assert model.compare(obs2, mod2)["n"] == 80
    with pytest.raises(ValueError):
        model.compare(obs[:20], mod[:20])


def test_correlation_interval_widens_with_autocorrelation():
    rng = np.random.default_rng(2)
    n = 1500
    x = rng.normal(size=n)
    iid = model.compare(x + 5, x + rng.normal(0, 1, n) + 5)
    auto_noise = synth.ar1(n, 0.95, 1.0, rng)
    slow = model.compare(synth.ar1(n, 0.95, 1.0, rng) + 5, synth.ar1(n, 0.95, 1.0, rng) + 5)
    assert iid["r_low"] < iid["r"] < iid["r_high"] and slow["n_eff"] < 0.2 * slow["n"] < iid["n_eff"]
    assert (slow["r_high"] - slow["r_low"]) > (iid["r_high"] - iid["r_low"])
    del auto_noise


def test_lag1_autocorr_and_effective_n():
    rng = np.random.default_rng(3)
    x = synth.ar1(20000, 0.8, 1.0, rng)
    assert model.lag1_autocorr(x) == pytest.approx(0.8, abs=0.02) and model.lag1_autocorr(np.arange(5.0)) == 0.0
    assert model.effective_n(x, x) < 0.5 * len(x)
    white = rng.normal(size=5000)
    assert model.effective_n(white, rng.normal(size=5000)) > 0.8 * 5000


def test_real_model_comparison(results):
    m = results.summary["model"]
    nmb = [v["NO2"]["nmb"] for v in m.values() if "NO2" in v]
    assert len(nmb) == 13 and sum(b < 0 for b in nmb) >= 11 and min(nmb) < -0.5
    o3 = [v["O3"]["nmb"] for v in m.values() if "O3" in v]
    assert len(o3) >= 8 and all(b > 0 for b in o3)
    assert all(v["PM10"]["r_low"] < v["PM10"]["r"] < v["PM10"]["r_high"] and v["PM10"]["r"] > 0.35 for v in m.values())

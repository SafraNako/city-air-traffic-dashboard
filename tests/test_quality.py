import numpy as np

from sehir_panosu import quality, synth


def test_completeness():
    x = np.array([1.0, np.nan, 3.0, np.nan])
    assert quality.completeness(x) == 0.5 and np.isnan(quality.completeness([]))
    years = np.array([2020, 2020, 2021, 2021])
    assert quality.completeness_by_year(years, x) == {2020: 0.5, 2021: 0.5}
    assert quality.completeness_by_year(years, np.array([1.0, 2.0, np.nan, np.nan])) == {2020: 1.0, 2021: 0.0}


def test_stuck_mask_runs_and_gaps():
    x = np.array([1.0, 2.0, 5.0, 5.0, 5.0, 5.0, 3.0, 4.0, 4.0, np.nan, 4.0, 4.0, 4.0, 4.0, 4.0, 7.0])
    m = quality.stuck_mask(x, 4)
    assert m.tolist() == [False, False, True, True, True, True, False, False, False, False, True, True, True, True, True, False]
    assert not quality.stuck_mask(x, 6).any() and quality.stuck_mask(x, 2)[7] and len(quality.stuck_mask(np.array([]), 3)) == 0


def test_repeat_probability_and_threshold():
    rng = np.random.default_rng(0)
    x = np.round(rng.normal(30, 10, 20000), 1)
    q = quality.repeat_probability(x)
    assert 0.001 < q < 0.006                                            # about 1 / (sd sqrt(4 pi) / 0.1) for a normal rounded to 0.1
    k = quality.stuck_threshold(x)
    assert k >= 4 and 20000 * q ** (k - 1) <= 0.5 + 1e-9 and (k == 4 or 20000 * q ** (k - 2) > 0.5)
    assert np.isnan(quality.repeat_probability(np.arange(10.0)))
    y = x.copy()
    y[5000:5040] = 33.3
    assert quality.stuck_threshold(y) == k                              # the planted run does not inflate its own threshold


def test_spike_mask_isolates_single_hours_but_not_episodes():
    rng = np.random.default_rng(1)
    x = 30 + rng.normal(0, 2, 400)
    x[100] = 400.0
    x[200:206] = 250.0                                                   # a six-hour episode: neighbours are high as well
    m = quality.spike_mask(x)
    assert m[100] and not m[200:206].any() and m.sum() <= 3


def test_physical_range():
    m = quality.physical_mask("NO2", np.array([-1.0, 0.0, 50.0, 1499.0, 1501.0, np.nan]))
    assert m.tolist() == [True, False, False, False, True, False]


def test_flag_series_and_clean():
    rng = np.random.default_rng(2)
    h = synth.make_series(200, rng)
    x = np.round(h["x"], 1)
    f, stuck, spike = synth.plant_faults(x, rng, stuck_runs=2, spikes=5)
    flags, run = quality.flag_series("NO2", f)
    assert set(flags) == {"stuck", "spike", "impossible"} and run >= 4
    cleaned, _, _ = quality.clean("NO2", f)
    assert np.isnan(cleaned[stuck]).all() and np.isnan(cleaned[spike]).all() and np.isfinite(cleaned[~(flags["stuck"] | flags["spike"])]).all()


def test_planted_faults_are_found():
    rng = np.random.default_rng(3)
    tp_s = n_s = fp_s = tp_p = n_p = fp_p = 0
    for _ in range(4):
        h = synth.make_series(730, rng, sigma=0.35, phi=0.9)
        f, stuck, spike = synth.plant_faults(np.round(h["x"], 1), rng)
        flags, _ = quality.flag_series("NO2", f)
        tp_s += int((flags["stuck"] & stuck).sum())
        n_s += int(stuck.sum())
        fp_s += int((flags["stuck"] & ~stuck).sum())
        tp_p += int((flags["spike"] & spike).sum())
        n_p += int(spike.sum())
        fp_p += int((flags["spike"] & ~spike).sum())
    assert tp_s / n_s > 0.98 and fp_s <= 6 and tp_p / n_p > 0.9 and fp_p == 0


def test_real_station_flags_are_a_small_share(ds):
    x = ds.air["aksaray"]["NO2"]
    flags, run = quality.flag_series("NO2", x)
    share = (flags["stuck"] | flags["spike"]).sum() / np.isfinite(x).sum()
    assert 0 < share < 0.02 and run >= 4 and not flags["impossible"].any()
    neg = ds.air["avcilar"]["NO2"]
    assert quality.physical_mask("NO2", neg).sum() == 8                  # eight negative readings in the real record

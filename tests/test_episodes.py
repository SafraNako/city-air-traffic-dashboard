import math

import numpy as np
import pytest

from sehir_panosu import episodes, special


def test_regional_median_needs_enough_stations():
    stack = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan], [5.0, 4.0, 9.0]])
    med, cnt = episodes.regional_median(stack, min_stations=2)
    assert med[0] == 3.0 and med[1] == 3.0 and np.isnan(med[2]) and cnt.tolist() == [3, 2, 1]


def test_find_episodes_runs_gaps_and_minimum_length():
    dates = np.datetime64("2020-01-01") + np.arange(15)
    x = np.array([10, 60, 70, 55, 10, 10, 80, 10, 90, 95, 10, 10, 10, 10, 65.0])
    eps = episodes.find_episodes(dates, x, 50.0)
    assert [(str(e["start"]), e["days"]) for e in eps] == [("2020-01-02", 3), ("2020-01-07", 1), ("2020-01-09", 2), ("2020-01-15", 1)]
    assert eps[0]["peak"] == 70 and eps[0]["mean"] == pytest.approx(185 / 3)
    merged = episodes.find_episodes(dates, x, 50.0, max_gap=1)
    assert [(str(e["start"]), e["days"]) for e in merged] == [("2020-01-02", 3), ("2020-01-07", 4), ("2020-01-15", 1)]
    assert [e["days"] for e in episodes.find_episodes(dates, x, 50.0, min_days=2)] == [3, 2]
    assert episodes.find_episodes(dates, np.full(15, np.nan), 50.0) == []


def test_first_component_share_detects_a_common_factor():
    rng = np.random.default_rng(0)
    n = 800
    common = rng.normal(size=n)
    stack = np.exp(3.0 + 0.6 * common[None, :] + 0.3 * rng.normal(size=(6, n)))
    share, days = episodes.first_component_share(stack)
    assert 0.7 < share < 0.85 and days == n
    indep = np.exp(3.0 + rng.normal(size=(6, n)))
    assert episodes.first_component_share(indep)[0] < 0.3
    holes = stack.copy()
    holes[0, :700] = np.nan
    assert episodes.first_component_share(holes, min_days=50)[1] == 100
    with pytest.raises(ValueError):
        episodes.first_component_share(holes, min_days=200)


def test_fisher_exact_against_the_hypergeometric_by_hand():
    assert episodes.fisher_exact_two_sided(3, 1, 1, 3) == pytest.approx(34 / 70)               # the tea-tasting table
    assert episodes.fisher_exact_two_sided(5, 0, 0, 5) == pytest.approx(2 / 252)
    assert episodes.fisher_exact_two_sided(10, 10, 10, 10) == pytest.approx(1.0)
    assert 0 < episodes.fisher_exact_two_sided(8, 2, 1, 9) < 0.02
    p = episodes.fisher_exact_two_sided(81, 325, 73, 2364)
    assert p < 1e-20
    a, b, c, d = 4, 6, 7, 3
    tables = [(x, 10 - x, 11 - x, 10 - (11 - x)) for x in range(1, 11)]
    probs = [math.comb(10, x) * math.comb(10, 11 - x) / math.comb(20, 11) for x in range(1, 11)]
    assert sum(probs) == pytest.approx(1.0)
    obs = math.comb(10, a) * math.comb(10, 11 - a) / math.comb(20, 11)
    assert episodes.fisher_exact_two_sided(a, b, c, d) == pytest.approx(sum(q for q in probs if q <= obs + 1e-12))
    assert len(tables) == 10


def test_overlap_table():
    ep = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    sig = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], bool)
    t = episodes.overlap_table(ep, sig)
    assert (t["both"], t["episode_only"], t["signal_only"], t["neither"]) == (2, 1, 1, 6) and t["odds_ratio"] == pytest.approx(12.0)
    assert t["hit_rate"] == pytest.approx(2 / 3) and t["precision"] == pytest.approx(2 / 3) and t["p"] == pytest.approx(episodes.fisher_exact_two_sided(2, 1, 1, 6))
    assert episodes.overlap_table(np.array([1, 0], bool), np.array([1, 0], bool))["odds_ratio"] == float("inf")


def test_rayleigh_test():
    rng = np.random.default_rng(1)
    mean, r, p, z = episodes.rayleigh(rng.normal(150, 30, 80) % 360)
    assert abs(mean - 150) < 15 and r > 0.8 and p < 1e-20 and z > 40
    ps = [episodes.rayleigh(rng.uniform(0, 360, 50))[2] for _ in range(300)]
    assert special.ks_uniform(np.array(ps))[1] > 0.001
    north = episodes.rayleigh(np.array([350.0, 10.0, 355.0, 5.0, 0.0]))[0]
    assert north < 1.5 or north > 358.5                                                   # the mean of directions around north is north
    with pytest.raises(ValueError):
        episodes.rayleigh(np.array([1.0, 2.0]))

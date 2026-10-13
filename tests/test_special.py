import math

import numpy as np
import pytest

from sehir_panosu import special


def test_chi2_two_degrees_is_exponential():
    for x in (0.1, 1.0, 5.0, 12.0):
        assert special.chi2_sf(x, 2) == pytest.approx(math.exp(-x / 2), rel=1e-12)
        assert special.chi2_cdf(x, 2) == pytest.approx(1 - math.exp(-x / 2), rel=1e-12)


def test_chi2_tabulated_critical_values():
    assert special.chi2_sf(3.841459, 1) == pytest.approx(0.05, abs=1e-6)
    assert special.chi2_sf(18.307038, 10) == pytest.approx(0.05, abs=1e-6)
    assert special.chi2_sf(124.342, 100) == pytest.approx(0.05, abs=2e-5)


def test_gammainc_complements_and_edges():
    for a, x in ((0.5, 0.3), (3.0, 2.0), (50.0, 60.0), (500.5, 520.0)):
        assert special.gammainc(a, x) + special.gammaincc(a, x) == pytest.approx(1.0, abs=1e-12)
    assert special.gammainc(2.0, 0.0) == 0.0
    assert special.gammaincc(2.0, -1.0) == 1.0
    with pytest.raises(ValueError):
        special.gammainc(0.0, 1.0)


def test_gamma_ppf_inverts_cdf_and_known_median():
    assert special.gamma_ppf(0.5, 1.0) == pytest.approx(math.log(2), rel=1e-10)
    for a in (0.7, 3.0, 40.0, 700.0):
        for q in (0.025, 0.5, 0.975):
            assert special.gammainc(a, special.gamma_ppf(q, a)) == pytest.approx(q, abs=1e-9)
    with pytest.raises(ValueError):
        special.gamma_ppf(1.0, 2.0)


def test_normal_functions():
    assert special.norm_cdf(0.0) == 0.5
    assert special.norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert special.norm_ppf(special.norm_cdf(-1.3)) == pytest.approx(-1.3, abs=1e-9)
    assert special.norm_sf(1.0) + special.norm_cdf(1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        special.norm_ppf(0.0)


@pytest.mark.parametrize("k,mu", [(0, 0.7), (3, 2.5), (12, 9.0), (40, 35.0)])
def test_poisson_cdf_against_direct_sum(k, mu):
    direct = sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(k + 1))
    assert special.poisson_cdf(k, mu) == pytest.approx(direct, rel=1e-10)
    assert special.poisson_sf(k, mu) == pytest.approx(1 - direct, abs=1e-12)


def test_poisson_edges():
    assert special.poisson_cdf(-1, 2.0) == 0.0
    assert special.poisson_cdf(3, 0.0) == 1.0
    assert special.poisson_sf(3, 0.0) == 0.0


def test_poisson_two_sided_p():
    assert special.poisson_two_sided_p(10, 10.0) == 1.0
    assert special.poisson_two_sided_p(0, 10.0) == pytest.approx(2 * math.exp(-10), rel=1e-10)
    assert special.poisson_two_sided_p(25, 10.0) < 1e-3
    assert 0 < special.poisson_two_sided_p(3, 10.0) < 0.1


def test_garwood_interval_tabulated():
    assert special.poisson_interval(0) == pytest.approx((0.0, 3.688879), abs=1e-5)
    lo, hi = special.poisson_interval(10)
    assert (lo, hi) == pytest.approx((4.795389, 18.390356), abs=1e-5)
    lo, hi = special.poisson_interval(10, level=0.90)
    assert 5.4 < lo < 5.5 and 16.9 < hi < 17.0


def test_garwood_coverage_is_at_least_nominal():
    for mu in (2.0, 7.5, 20.0):
        covered = 0.0
        for k in range(0, 120):
            lo, hi = special.poisson_interval(k)
            if lo <= mu <= hi:
                covered += math.exp(-mu) * mu ** k / math.factorial(k)
        assert covered >= 0.95 - 1e-9


def test_binomial_exact():
    assert special.binom_cdf(3, 10, 0.5) == pytest.approx(0.171875)
    assert special.binom_cdf(-1, 10, 0.5) == 0.0
    assert special.binom_cdf(10, 10, 0.5) == 1.0
    assert special.binom_two_sided_p(3, 10, 0.5) == pytest.approx(0.34375)
    assert special.binom_two_sided_p(5, 10, 0.5) == pytest.approx(1.0)
    assert special.binom_pmf_array(5, 0.0)[0] == 1.0 and special.binom_pmf_array(5, 1.0)[5] == 1.0
    assert special.binom_pmf_array(30, 0.3).sum() == pytest.approx(1.0)


def test_betainc_and_clopper_pearson():
    assert special.betainc(0.3, 2, 5) == pytest.approx(0.579825, abs=1e-6)
    assert special.betainc(0.0, 2, 5) == 0.0 and special.betainc(1.0, 2, 5) == 1.0
    assert special.betainc(0.5, 3, 3) == pytest.approx(0.5, abs=1e-12)
    assert special.clopper_pearson(0, 10) == pytest.approx((0.0, 0.308497), abs=1e-5)
    assert special.clopper_pearson(10, 10)[1] == 1.0
    assert special.clopper_pearson(5, 10) == pytest.approx((0.187086, 0.812914), abs=1e-5)
    lo, hi = special.clopper_pearson(30, 200)
    assert lo < 0.15 < hi


def test_clopper_pearson_coverage_at_least_nominal():
    n, p = 40, 0.3
    covered = sum(special.binom_pmf_array(n, p)[k] for k in range(n + 1) if special.clopper_pearson(k, n)[0] <= p <= special.clopper_pearson(k, n)[1])
    assert covered >= 0.95 - 1e-9


def test_ks_uniform_sample_and_counterexamples():
    rng = np.random.default_rng(0)
    d, p = special.ks_uniform(rng.random(2000))
    assert d < 0.05 and p > 0.01
    d, p = special.ks_uniform(rng.random(2000) ** 3)
    assert p < 1e-6
    d, p = special.ks_uniform(np.array([0.5]))
    assert d == pytest.approx(0.5)
    with pytest.raises(ValueError):
        special.ks_uniform([])


def test_student_t_distribution():
    assert special.t_cdf(0.0, 5) == 0.5
    assert special.t_cdf(2.0, 1) == pytest.approx(0.852416, abs=1e-6)            # Cauchy: 1/2 + atan(2)/pi
    assert special.t_cdf(2.0, 1) == pytest.approx(0.5 + math.atan(2.0) / math.pi, abs=1e-12)
    assert special.t_ppf(0.975, 10) == pytest.approx(2.228139, abs=1e-5)
    assert special.t_ppf(0.975, 100) == pytest.approx(1.983972, abs=1e-5)
    assert special.t_ppf(0.975, 100000) == pytest.approx(1.959964, abs=1e-3)
    for df in (3, 12, 60):
        for q in (0.05, 0.5, 0.9):
            assert special.t_cdf(special.t_ppf(q, df), df) == pytest.approx(q, abs=1e-9)
    assert special.t_sf(1.5, 7) == pytest.approx(1 - special.t_cdf(1.5, 7))
    assert special.t_cdf(-1.5, 7) == pytest.approx(special.t_sf(1.5, 7), abs=1e-12)
    with pytest.raises(ValueError):
        special.t_cdf(1.0, 0)
    with pytest.raises(ValueError):
        special.t_ppf(1.0, 5)


def test_normal_two_sided_p():
    assert special.normal_two_sided_p(1.959964) == pytest.approx(0.05, abs=1e-6)
    assert special.normal_two_sided_p(0.0) == pytest.approx(1.0)

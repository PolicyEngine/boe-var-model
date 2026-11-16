import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boe_var import unemployment_satellite as satellite
from boe_var.data import load_data
from boe_var.forecast import yoy


def _default_fit():
    df = load_data()
    g = yoy(np.asarray(df["uk_gdp"], dtype=float))
    gq = [str(q) for q in df.index][4:]
    return satellite.fit_default(gq, g)


def test_fit_recovers_negative_growth_coefficient():
    fit = _default_fit()
    assert fit.beta < 0, "growth coefficient must be negative on UK data"
    assert fit.nobs > 100
    assert fit.sample == ("1992Q1", "2025Q1")


def test_forecast_monotone_decreasing_in_gdp_growth():
    fit = _default_fit()
    low = fit.forecast(np.full(8, 0.0), u_last=5.0, du_last=0.0)
    high = fit.forecast(np.full(8, 3.0), u_last=5.0, du_last=0.0)
    assert (high < low).all()


def test_band_mapping_swaps_bounds_and_orders_correctly():
    fit = _default_fit()
    gdp = {
        "median": np.full(8, 1.5),
        "lo68": np.full(8, 0.5), "hi68": np.full(8, 2.5),
        "lo90": np.full(8, -0.5), "hi90": np.full(8, 3.5),
    }
    bands = satellite.unemployment_bands(fit, gdp, u_last=5.0, du_last=0.0)
    assert (bands["lo90"] <= bands["lo68"]).all()
    assert (bands["lo68"] <= bands["median"]).all()
    assert (bands["median"] <= bands["hi68"]).all()
    assert (bands["hi68"] <= bands["hi90"]).all()


def test_synthetic_recovery():
    rng = np.random.default_rng(0)
    n = 400
    g = rng.normal(2.0, 1.5, n)
    du = np.empty(n)
    du_prev = 0.0
    for t in range(n):
        du[t] = 0.05 - 0.3 * g[t] + 0.4 * du_prev + rng.normal(0, 0.01)
        du_prev = du[t]
    fit = satellite.fit_satellite(g[1:], du[1:], du[:-1])
    assert abs(fit.beta - (-0.3)) < 0.02
    assert abs(fit.gamma - 0.4) < 0.05


# --- joint predictive bands -------------------------------------------------

def _gdp_paths(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    return 1.5 + rng.standard_normal((n, 4)) * np.array([0.6, 0.9, 1.1, 1.2])


def test_predictive_bands_are_reproducible_with_a_seed():
    fit = _default_fit()
    g = _gdp_paths()
    a = satellite.unemployment_predictive_bands(
        fit, g, 5.0, 0.0, np.random.default_rng(3))
    b = satellite.unemployment_predictive_bands(
        fit, g, 5.0, 0.0, np.random.default_rng(3))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_no_residual_and_no_coefficient_draws_reproduce_the_forecast():
    fit = _default_fit()
    g = np.array([1.0, 0.5, -0.2, 2.0])
    paths = satellite.simulate_unemployment_paths(
        fit, g[None], 4.9, -0.1, np.random.default_rng(0),
        residual=False, coefficients=False)
    np.testing.assert_allclose(paths[0], fit.forecast(g, 4.9, -0.1),
                               rtol=0, atol=1e-12)


def test_h1_68_half_width_is_at_least_the_residual_sd():
    """One quarter of satellite residual alone is +/- sigma at 68%; the old
    GDP-only band at h1 was a fraction of that."""
    fit = _default_fit()
    g = np.full((4000, 4), 1.5)  # no GDP uncertainty at all
    b = satellite.unemployment_predictive_bands(
        fit, g, 5.0, 0.0, np.random.default_rng(0))
    half = 0.5 * (b["hi68"][0] - b["lo68"][0])
    assert half >= 0.95 * fit.sigma
    assert half <= 1.2 * fit.sigma


def test_joint_bands_are_no_narrower_than_gdp_only_bands():
    fit = _default_fit()
    g = _gdp_paths()
    from boe_var.analysis import aggregate
    old = satellite.unemployment_bands(fit, aggregate(list(g)), 5.0, 0.0)
    new = satellite.unemployment_predictive_bands(
        fit, g, 5.0, 0.0, np.random.default_rng(0))
    assert ((new["hi68"] - new["lo68"]) >= (old["hi68"] - old["lo68"])).all()
    assert ((new["hi90"] - new["lo90"]) >= (old["hi90"] - old["lo90"])).all()


def test_coefficient_covariance_is_the_ols_one():
    fit = _default_fit()
    assert fit.coef_cov.shape == (3, 3)
    np.testing.assert_allclose(fit.coef_cov, fit.coef_cov.T)
    assert (np.linalg.eigvalsh(fit.coef_cov) > 0).all()

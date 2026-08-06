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

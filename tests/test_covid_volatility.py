"""Lenza-Primiceri Covid volatility scaling (boe_var.covid_volatility).

Pins: scale = 1 reproduces the plain BVAR and the default coverage run bit
for bit; the scale estimate reacts to an outlier and not to calm data; the
real-time evaluation uses nothing after the origin.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boe_var import covid_volatility as cv
from boe_var import forecast as fc
from boe_var.bvar import BVAR
from boe_var.data import load_data

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    from run_coverage_evaluation import run_coverage  # noqa: E402
    from run_covid_volatility_evaluation import run_coverage_lp  # noqa: E402
finally:
    sys.path.pop(0)


def _var1(T=120, k=3, seed=0, start="1990Q1"):
    rng = np.random.default_rng(seed)
    A = np.diag([0.6, 0.5, 0.4])[:k, :k]
    y = np.zeros((T, k))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + rng.standard_normal(k)
    return y, pd.period_range(start, periods=T, freq="Q")


def test_beta_prior_has_the_stated_mode_and_sd():
    a, b = cv.RHO_PRIOR
    assert (a - 1) / (a + b - 2) == pytest.approx(0.8)
    assert np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))) == pytest.approx(0.2)


def test_scale_path_follows_the_lp_law_of_motion():
    idx = pd.period_range("2019Q4", "2021Q4", freq="Q")
    p = {"s0": 10.0, "s1": 6.0, "s2": 3.0, "rho": 0.5}
    s = cv.scale_path(idx, p)
    # 2019Q4, 2020Q1 | 2020Q2-Q4 free | 2021Q1.. = 1 + 2 * 0.5^j
    np.testing.assert_allclose(s, [1, 1, 10, 6, 3, 2, 1.5, 1.25, 1.125])
    # forecast from 2020Q4 continues the same path
    np.testing.assert_allclose(cv.future_scale(idx[:5], p, 4), s[5:9])
    # before the start every future scale is exactly one
    assert (cv.future_scale(idx[:2], p, 3) == 1.0).all()


def test_unit_scale_reproduces_the_plain_bvar_exactly():
    y = load_data().to_numpy(float)
    for kw in ({}, {"theta": 1.0}):
        a = BVAR(y, lags=4, **kw)
        b = cv.ScaledBVAR(y, lags=4, scale=np.ones(len(y)), **kw)
        assert np.array_equal(a.B_post, b.B_post)
        assert np.array_equal(a.S_post, b.S_post)
        assert a.df_post == b.df_post
        assert a.log_marginal_likelihood() == b.log_marginal_likelihood()
        da, db = a.sample_posterior(5, seed=3), b.sample_posterior(5, seed=3)
        for x, z in zip(da, db):
            assert np.array_equal(x.Pi, z.Pi)
            assert np.array_equal(x.Sigma, z.Sigma)


def test_unit_shock_scale_reproduces_sample_forecast_exactly():
    y = load_data().to_numpy(float)
    d = BVAR(y, lags=4).sample_posterior(1, seed=0)[0]
    a = fc.sample_forecast(d, y, horizons=6, rng=np.random.default_rng(9))
    b = cv.sample_forecast_scaled(d, y, horizons=6,
                                  rng=np.random.default_rng(9),
                                  shock_scale=np.ones(6))
    assert np.array_equal(a, b)


def test_unit_scale_coverage_run_reproduces_the_default_exactly():
    df = load_data()
    y = df.to_numpy(float)
    T = y.shape[0]
    kw = dict(lags=4, horizons=3, first_origin=T - 3 - 4, n_draws=8,
              n_paths=2, seed=1)
    ra, rb = {}, {}
    ha, na = run_coverage(y, record=ra, **kw)
    hb, nb = run_coverage_lp(y, df.index, estimate=False, record=rb, **kw)
    assert np.array_equal(na, nb)
    for lv in (68, 90):
        assert np.array_equal(ha[lv], hb[lv])
        assert np.array_equal(ra["stretch"][lv], rb["stretch"][lv])
    assert np.array_equal(ra["quantiles"], rb["quantiles"])


def test_pre_covid_origins_are_the_default_model():
    """An origin before 2020Q2 sees no Covid quarter, so the LP run must be
    the headline run there even with estimation switched on."""
    df = load_data()
    y = df.to_numpy(float)
    first = list(df.index).index(pd.Period("2019Q2", "Q"))
    kw = dict(lags=4, horizons=2, first_origin=first, n_draws=6, n_paths=2,
              seed=2)
    ra, rb = {}, {}
    run_coverage(y[: first + 3 + 2], record=ra, **kw)
    run_coverage_lp(y[: first + 3 + 2], df.index[: first + 5], record=rb,
                    **kw)
    assert np.array_equal(ra["quantiles"], rb["quantiles"])
    assert all(s["s0"] is None for s in rb["scales"])


def test_scale_estimate_finds_an_outlier_and_ignores_calm_data():
    y, idx = _var1(start="1990Q1", T=130)
    j = list(idx).index(pd.Period("2020Q2", "Q"))
    calm = cv.estimate_covid_scale(y[: j + 6], idx[: j + 6], lags=1)
    assert calm["estimated"] == ["s0", "s1", "s2", "rho"]
    assert max(calm["s0"], calm["s1"], calm["s2"]) < 2.0
    hit = y.copy()
    hit[j] += 15.0      # a 15-sd innovation in 2020Q2 ...
    hit[j + 1] -= 15.0  # ... reversed next quarter
    big = cv.estimate_covid_scale(hit[: j + 6], idx[: j + 6], lags=1)
    assert big["s0"] > 5.0
    assert big["log_post"] > -np.inf


def test_scale_estimate_only_identifies_what_the_sample_contains():
    y, idx = _var1(T=130)
    j = list(idx).index(pd.Period("2020Q2", "Q"))
    p = cv.estimate_covid_scale(y[: j + 2], idx[: j + 2], lags=1)
    assert p["estimated"] == ["s0", "s1"]
    assert p["s2"] == 1.0 and p["rho"] == cv.RHO_MODE
    none = cv.estimate_covid_scale(y[:j], idx[:j], lags=1)
    assert none["estimated"] == [] and none["s0"] == 1.0


def test_lp_coverage_run_uses_nothing_after_the_origin():
    """Perturbing data after an origin's last forecast outturn leaves that
    origin's scale estimate and band geometry unchanged."""
    df = load_data()
    y = df.to_numpy(float)
    o = list(df.index).index(pd.Period("2021Q1", "Q"))
    H = 2
    kw = dict(lags=4, horizons=H, first_origin=o, n_draws=6, n_paths=2,
              seed=4)
    ya = y[: o + H + 2]          # two origins: o and o + 1
    yb = ya.copy()
    yb[o + 1 + H] += 50.0        # last outturn of origin o + 1 only
    ra, rb = {}, {}
    run_coverage_lp(ya, df.index[: len(ya)], record=ra, **kw)
    run_coverage_lp(yb, df.index[: len(yb)], record=rb, **kw)
    assert ra["scales"][0] == rb["scales"][0]
    assert np.array_equal(ra["quantiles"][0], rb["quantiles"][0])
    # the origin-o + 1 estimate does see data through o + 1, which is
    # unchanged, so its bands are identical too; only its outturn moved
    assert ra["scales"][1] == rb["scales"][1]
    assert np.array_equal(ra["quantiles"][1], rb["quantiles"][1])
    assert not np.array_equal(ra["actual"][1], rb["actual"][1])


def test_lp_training_change_moves_later_origins_only():
    df = load_data()
    y = df.to_numpy(float)
    o = list(df.index).index(pd.Period("2021Q1", "Q"))
    H = 2
    kw = dict(lags=4, horizons=H, first_origin=o, n_draws=6, n_paths=2,
              seed=4)
    ya = y[: o + H + 2]
    yb = ya.copy()
    yb[o + 1] += 5.0             # in origin o + 1's training sample
    ra, rb = {}, {}
    run_coverage_lp(ya, df.index[: len(ya)], record=ra, **kw)
    run_coverage_lp(yb, df.index[: len(yb)], record=rb, **kw)
    assert ra["scales"][0] == rb["scales"][0]
    assert np.array_equal(ra["quantiles"][0], rb["quantiles"][0])
    assert ra["scales"][1] != rb["scales"][1]


def test_committed_artifact_is_not_stale():
    from boe_var.evaluation import evaluation_code_version

    path = Path(__file__).resolve().parents[1] / "results" / \
        "covid_volatility_evaluation.json"
    import json

    recorded = json.loads(path.read_text())["code_version"]
    current = evaluation_code_version(extra_paths=[
        cv.__file__, SCRIPTS / "run_covid_volatility_evaluation.py",
        SCRIPTS / "run_coverage_evaluation.py"])
    assert recorded == current, (
        "re-run scripts/run_covid_volatility_evaluation.py and commit "
        "results/covid_volatility_evaluation.json")

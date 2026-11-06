"""Common stochastic volatility BVAR (boe_var.common_sv, CCM 2016).

Pins: volatility fixed at 1 reproduces the homoskedastic BVAR, its forecasts
and the default coverage run bit for bit; the KSC mixture has the moments of
log chi2_1; the sampler finds a volatility break; the held-out evaluation
uses nothing after the origin.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boe_var import common_sv as sv
from boe_var import covid_volatility as cv
from boe_var import forecast as fc
from boe_var.bvar import BVAR
from boe_var.data import load_data

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RESULTS = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(SCRIPTS))
try:
    from run_common_sv_evaluation import covid_dummy_matrix, run_coverage_sv  # noqa: E402
    from run_coverage_evaluation import run_coverage  # noqa: E402
finally:
    sys.path.pop(0)

FAST = dict(burn=20, thin=1)


def test_ksc_mixture_has_the_moments_of_log_chi2_1():
    q, m, v = sv.KSC_PROBS, sv.KSC_MEANS, sv.KSC_VARS
    assert q.sum() == pytest.approx(1.0, abs=1e-6)
    mean = q @ m
    assert mean == pytest.approx(-1.2704, abs=1e-3)          # E log chi2_1
    assert q @ (v + m ** 2) - mean ** 2 == pytest.approx(np.pi ** 2 / 2,
                                                         abs=1e-2)


@pytest.mark.parametrize("kw", [{}, {"theta": 1.0}, {"dummies": True}])
def test_fixed_volatility_reproduces_the_homoskedastic_posterior(kw):
    df = load_data()
    y = df.to_numpy(float)
    kw = dict(kw)
    if kw.pop("dummies", False):
        kw["dummies"] = covid_dummy_matrix(df.index)
    a = BVAR(y, lags=4, **kw).sample_posterior(5, seed=3)
    b = sv.CommonSVBVAR(y, lags=4, **kw).sample(5, seed=3,
                                                fix_volatility=True)
    for x, z in zip(a, b):
        assert np.array_equal(x.Pi, z.Pi)
        assert np.array_equal(x.Sigma, z.Sigma)
        assert z.h_T == 0.0 and z.sigma2 == 0.0


def test_weighted_moments_match_the_lp_scaled_bvar():
    """Step 1 of the sampler is #21's ScaledBVAR with s_t = exp(h_t / 2)."""
    y = load_data().to_numpy(float)
    rng = np.random.default_rng(0)
    h = rng.normal(0, 1, size=len(y))
    m = sv.CommonSVBVAR(y, lags=4)
    C, B, S = m._moments(h[4:])
    ref = cv.ScaledBVAR(y, lags=4, scale=np.exp(0.5 * h))
    np.testing.assert_allclose(B, ref.B_post, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(S, ref.S_post, rtol=1e-10, atol=1e-12)


def test_zero_volatility_draw_forecasts_like_sample_forecast():
    y = load_data().to_numpy(float)
    d = sv.CommonSVBVAR(y, lags=4).sample(1, seed=0, fix_volatility=True)[0]
    a = fc.sample_forecast(d, y, horizons=6, rng=np.random.default_rng(9))
    b = sv.sample_forecast_sv(d, y, horizons=6, rng=np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_log_vol_law_of_motion():
    rng = np.random.default_rng(0)
    base = dict(Pi=np.zeros((1, 2)), Sigma=np.eye(1), lags=1, k=1)
    rw = sv.SVDraw(**base, h_T=0.7, sigma2=0.0, phi=1.0)
    np.testing.assert_allclose(sv.simulate_log_vol(rw, 4, rng), [0.7] * 4)
    ar = sv.SVDraw(**base, h_T=1.0, sigma2=0.0, phi=0.5)
    np.testing.assert_allclose(sv.simulate_log_vol(ar, 3, rng),
                               [0.5, 0.25, 0.125])
    # with an intercept h reverts to c / (1 - phi) = 2
    ar = sv.SVDraw(**base, h_T=1.0, sigma2=0.0, phi=0.5, c=1.0)
    np.testing.assert_allclose(sv.simulate_log_vol(ar, 3, rng),
                               [1.5, 1.75, 1.875])


@pytest.mark.parametrize("with_dummies", [False, True])
def test_fixed_volatility_coverage_run_reproduces_the_default(with_dummies):
    df = load_data()
    y = df.to_numpy(float)
    T = y.shape[0]
    D = covid_dummy_matrix(df.index) if with_dummies else None
    kw = dict(lags=4, horizons=3, first_origin=T - 3 - 4, n_draws=8,
              n_paths=2, seed=1, dummies=D)
    ra, rb = {}, {}
    ha, na = run_coverage(y, record=ra, **kw)
    hb, nb = run_coverage_sv(y, fix_volatility=True, record=rb, **kw)
    assert np.array_equal(na, nb)
    for lv in (68, 90):
        assert np.array_equal(ha[lv], hb[lv])
        assert np.array_equal(ra["stretch"][lv], rb["stretch"][lv])
    assert np.array_equal(ra["quantiles"], rb["quantiles"])


@pytest.mark.parametrize("law", ["rw", "ar1"])
def test_sampler_finds_a_volatility_break(law):
    rng = np.random.default_rng(0)
    T, k = 200, 3
    lam = np.where(np.arange(T) > 120, 9.0, 1.0)
    y = np.zeros((T, k))
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + np.sqrt(lam[t]) * rng.standard_normal(k)
    m = sv.CommonSVBVAR(y, lags=1, law=law)
    draws = m.sample(100, seed=1, burn=300, thin=3)
    gap = m.h_mean[130:].mean() - m.h_mean[10:110].mean()
    assert 1.4 < gap < 3.0                      # true log 9 = 2.20
    assert all(np.isfinite(d.Sigma).all() for d in draws)
    if law == "ar1":
        assert all(abs(d.phi) < 1 for d in draws)


def test_calm_data_give_flat_volatility():
    rng = np.random.default_rng(2)
    y = np.zeros((160, 3))
    for t in range(1, 160):
        y[t] = 0.5 * y[t - 1] + rng.standard_normal(3)
    m = sv.CommonSVBVAR(y, lags=1)
    m.sample(100, seed=1, burn=300, thin=3)
    assert np.ptp(m.h_mean) < 1.0


def _sv_run(y, idx_len, o, H):
    rec = {}
    run_coverage_sv(y, lags=4, horizons=H, first_origin=o, n_draws=6,
                    n_paths=2, seed=4, record=rec, **FAST)
    return rec


def test_sv_coverage_run_uses_nothing_after_the_origin():
    """Perturbing an outturn after origin o + 1 leaves both origins'
    volatility estimates and bands unchanged; only the outturn moves."""
    df = load_data()
    y = df.to_numpy(float)
    o = list(df.index).index(pd.Period("2021Q1", "Q"))
    H = 2
    ya = y[: o + H + 2]
    yb = ya.copy()
    yb[o + 1 + H] += 50.0
    ra, rb = _sv_run(ya, len(ya), o, H), _sv_run(yb, len(yb), o, H)
    assert ra["volatility"] == rb["volatility"]
    assert np.array_equal(ra["quantiles"], rb["quantiles"])
    assert not np.array_equal(ra["actual"][1], rb["actual"][1])


def test_sv_training_change_moves_later_origins_only():
    df = load_data()
    y = df.to_numpy(float)
    o = list(df.index).index(pd.Period("2021Q1", "Q"))
    H = 2
    ya = y[: o + H + 2]
    yb = ya.copy()
    yb[o + 1] += 5.0             # in origin o + 1's training sample only
    ra, rb = _sv_run(ya, len(ya), o, H), _sv_run(yb, len(yb), o, H)
    assert ra["volatility"][0] == rb["volatility"][0]
    assert np.array_equal(ra["quantiles"][0], rb["quantiles"][0])
    assert ra["volatility"][1] != rb["volatility"][1]


@pytest.mark.parametrize("name,extra", [
    ("common_sv_evaluation.json", ["run_common_sv_evaluation.py",
                                   "run_coverage_evaluation.py",
                                   "run_covid_volatility_evaluation.py"]),
    ("explosive_roots_sweep.json", ["run_explosive_roots_sweep.py",
                                    "run_common_sv_evaluation.py",
                                    "run_coverage_evaluation.py",
                                    "run_covid_volatility_evaluation.py"]),
])
def test_committed_artifact_is_not_stale(name, extra):
    from boe_var.evaluation import evaluation_code_version

    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    recorded = json.loads(path.read_text())["code_version"]
    current = evaluation_code_version(
        extra_paths=[sv.__file__] + [SCRIPTS / e for e in extra])
    assert recorded == current, f"re-run the script that writes {name}"

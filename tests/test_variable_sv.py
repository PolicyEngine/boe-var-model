"""Variable-specific stochastic volatility BVAR (boe_var.variable_sv).

Pins: with every volatility fixed at 1 the conditional draws are the
conjugate BVAR's and the Gibbs sampler reproduces its NIW posterior; the
sampler finds a break in one variable without moving the others; the
held-out evaluation uses nothing after the origin; the draws plug into the
structural identification at any date.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import invwishart

from boe_var import forecast as fc
from boe_var import variable_sv as vs
from boe_var.bvar import BVAR
from boe_var.data import load_data

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RESULTS = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(SCRIPTS))
try:
    from run_common_sv_evaluation import covid_dummy_matrix  # noqa: E402
    from run_variable_sv_evaluation import run_coverage_vsv  # noqa: E402
finally:
    sys.path.pop(0)

FAST = dict(burn=20, thin=1)


def _factors(Sigma):
    L = np.linalg.cholesky(Sigma)
    d = np.diag(L) ** 2
    return np.linalg.inv(L / np.diag(L)), d


def test_factors_round_trip_and_at():
    y = load_data().to_numpy(float)
    S = BVAR(y, lags=4).sample_posterior(1, seed=0)[0].Sigma
    A, d = _factors(S)
    np.testing.assert_allclose(vs.sigma_from_factors(A, d), S, rtol=1e-10,
                               atol=1e-12)
    draw = vs.VSVDraw(Pi=np.zeros((8, 33)), Sigma=S, lags=4, k=8, A=A, d=d,
                      h_T=np.zeros(8), sigma2=np.zeros(8))
    np.testing.assert_allclose(draw.at(np.zeros(8)).Sigma, S, rtol=1e-10,
                               atol=1e-12)
    h = np.log(np.arange(1.0, 9.0))
    St = draw.at(h).Sigma
    np.testing.assert_allclose(St, vs.sigma_from_factors(A, d * np.exp(h)))
    assert np.allclose(draw.Sigma, S)          # at() does not mutate


@pytest.mark.parametrize("kw", [{}, {"theta": 1.0}, {"dummies": True}])
def test_unit_volatility_conditional_is_the_conjugate_one(kw):
    """h = 0: Pi | Sigma has mean B_post and precision kron(Sigma^-1, X*'X*),
    exactly the conditional of BVAR's matric-normal-inverse-Wishart."""
    df = load_data()
    y = df.to_numpy(float)
    kw = dict(kw)
    if kw.pop("dummies", False):
        kw["dummies"] = covid_dummy_matrix(df.index)
    m = vs.VariableSVBVAR(y, lags=4, **kw)
    S = m.base.sample_posterior(1, seed=1)[0].Sigma
    A, d = _factors(S)
    C, mean = m._pi_moments(A, d, np.zeros((m.n_obs, m.k)))
    np.testing.assert_allclose(mean, m.base.B_post, rtol=1e-6, atol=1e-8)
    XtX = m.base._Xs.T @ m.base._Xs
    C = np.tril(C)        # scipy leaves the unused upper triangle unset
    np.testing.assert_allclose(C @ C.T, np.kron(np.linalg.inv(S), XtX),
                               rtol=1e-8, atol=1e-6)


def test_unit_volatility_covariance_step_is_inverse_wishart():
    """h = 0: the triangular (A, d) draw given Pi is IW(U*'U*, T*)."""
    y = load_data().to_numpy(float)
    m = vs.VariableSVBVAR(y, lags=4)
    Pi = m.base.B_post.T
    U = m.base._Ys - m.base._Xs @ Pi.T
    rng = np.random.default_rng(0)
    h = np.zeros((m.n_obs, m.k))
    draws = np.array([vs.sigma_from_factors(*m._draw_A_d(Pi, h, rng))
                      for _ in range(4000)])
    ref = invwishart.rvs(df=m.n_rows, scale=U.T @ U, size=4000,
                         random_state=1)
    sd = ref.std(axis=0)
    assert np.abs((draws.mean(0) - ref.mean(0)) / sd).max() < 0.15
    np.testing.assert_allclose(draws.std(0), sd, rtol=0.12)


def test_unit_volatility_gibbs_reproduces_the_homoskedastic_posterior():
    y = load_data().to_numpy(float)
    m = vs.VariableSVBVAR(y, lags=4)
    draws = m.sample(400, seed=1, burn=100, thin=2, fix_volatility=True)
    ref = m.base.sample_posterior(4000, seed=2)
    for f in (lambda d: d.Pi, lambda d: d.Sigma):
        a = np.array([f(d) for d in draws])
        r = np.array([f(d) for d in ref])
        sd = r.std(0) + 1e-12
        assert np.abs((a.mean(0) - r.mean(0)) / sd).max() < 0.3
        assert 0.85 < np.median(a.std(0) / sd) < 1.15
    for d in draws:
        assert np.array_equal(d.h_T, np.zeros(m.k))
        assert np.array_equal(d.sigma2, np.zeros(m.k))
    assert np.array_equal(m.h_mean, np.zeros((m.n_obs, m.k)))


def test_unit_volatility_forecast_shocks_have_covariance_sigma():
    y = load_data().to_numpy(float)
    d = vs.VariableSVBVAR(y, lags=4).sample(1, seed=0, fix_volatility=True,
                                            **FAST)[0]
    rng = np.random.default_rng(9)
    path = vs.sample_forecast_vsv(d, y, horizons=5, rng=rng)
    z = np.random.default_rng(9).standard_normal((5, d.k))
    shocks = z @ (np.linalg.inv(d.A) * np.sqrt(d.d)).T
    np.testing.assert_allclose(path, fc._iterate(d, y, 5, shocks=shocks),
                               rtol=1e-10, atol=1e-10)


def test_forecast_volatility_random_walk():
    k = 2
    d = vs.VSVDraw(Pi=np.zeros((k, k + 1)), Sigma=np.eye(k), lags=1, k=k,
                   A=np.eye(k), d=np.ones(k), h_T=np.log([4.0, 1.0]),
                   sigma2=np.zeros(k))
    paths = np.array([vs.sample_forecast_vsv(d, np.zeros((1, k)), 3,
                                             np.random.default_rng(s))
                      for s in range(4000)])
    np.testing.assert_allclose(paths.std(axis=0), [[2.0, 1.0]] * 3,
                               rtol=0.06)


def test_sampler_finds_a_break_in_one_variable_only():
    rng = np.random.default_rng(0)
    T, k = 220, 3
    lam = np.where(np.arange(T) > 120, 9.0, 1.0)
    y = np.zeros((T, k))
    for t in range(1, T):
        e = rng.standard_normal(k)
        e[1] *= np.sqrt(lam[t])
        y[t] = 0.5 * y[t - 1] + e
    m = vs.VariableSVBVAR(y, lags=1)
    m.sample(100, seed=1, burn=300, thin=3)
    gap = m.h_mean[135:].mean(0) - m.h_mean[10:110].mean(0)
    assert 1.4 < gap[1] < 3.0                   # true log 9 = 2.20
    assert abs(gap[0]) < 0.6 and abs(gap[2]) < 0.6


# Bands from two runs on the same training data agree to the last few bits,
# not bit for bit, on some x86 CPUs: OpenBLAS's Sapphire Rapids kernels
# round differently with buffer alignment, which depends on what ran earlier
# in the process (measured on GitHub Actions: max |diff| 1e-7 on bands of
# order 1e3, about 1e-10 relative; bit-exact with OPENBLAS_CORETYPE=Haswell).
# A leak of the +50 / +5 perturbations would move the bands by far more than
# this tolerance.
BAND_RTOL = 1e-9


def assert_same_bands(a, b):
    np.testing.assert_allclose(a, b, rtol=BAND_RTOL, atol=0.0)


def _vsv_run(y, o, H):
    rec = {}
    run_coverage_vsv(y, lags=4, horizons=H, first_origin=o, n_draws=6,
                     n_paths=2, seed=4, record=rec, **FAST)
    return rec


def test_vsv_coverage_run_uses_nothing_after_the_origin():
    df = load_data()
    y = df.to_numpy(float)
    o = list(df.index).index(pd.Period("2021Q1", "Q"))
    H = 2
    ya = y[: o + H + 2]
    yb = ya.copy()
    yb[o + 1 + H] += 50.0
    ra, rb = _vsv_run(ya, o, H), _vsv_run(yb, o, H)
    assert ra["volatility"] == rb["volatility"]
    assert_same_bands(ra["quantiles"], rb["quantiles"])
    assert not np.array_equal(ra["actual"][1], rb["actual"][1])


def test_vsv_training_change_moves_later_origins_only():
    df = load_data()
    y = df.to_numpy(float)
    o = list(df.index).index(pd.Period("2021Q1", "Q"))
    H = 2
    ya = y[: o + H + 2]
    yb = ya.copy()
    yb[o + 1] += 5.0             # in origin o + 1's training sample only
    ra, rb = _vsv_run(ya, o, H), _vsv_run(yb, o, H)
    assert ra["volatility"][0] == rb["volatility"][0]
    assert_same_bands(ra["quantiles"][0], rb["quantiles"][0])
    assert ra["volatility"][1] != rb["volatility"][1]


def test_vsv_draws_identify_at_any_date():
    """Sigma_t = A^-1 D_t A^-1' is not a multiple of Sigma, so the impact
    matrix must be found at each date; the zero restrictions then hold on
    B_t = chol(Sigma_t) Q_t, and B_t is not a rescaling of B_ref."""
    from boe_var import identification as ident

    y = load_data().to_numpy(float)
    draw = vs.VariableSVBVAR(y, lags=4).sample(1, seed=5, **FAST)[0]
    rng = np.random.default_rng(0)
    h = np.log(np.array([1.0, 1.0, 9.0, 1.0, 2.0, 0.5, 4.0, 1.0]))
    Bs = []
    for dr in (draw, draw.at(h)):
        Q = ident.draw_Q(dr.Sigma, rng)
        B = np.linalg.cholesky(dr.Sigma) @ Q
        np.testing.assert_allclose(B @ B.T, dr.Sigma, rtol=1e-8, atol=1e-10)
        for shock, rows in ident.ZERO_RESTRICTIONS.items():
            j = ident.SHOCKS.index(shock)
            assert np.allclose(B[rows, j], 0.0, atol=1e-8)
        Bs.append(B)
    St = draw.at(h).Sigma
    lam = np.trace(St) / np.trace(draw.Sigma)
    assert not np.allclose(St, lam * draw.Sigma, rtol=0.05)


def test_committed_artifact_is_not_stale():
    from boe_var.evaluation import evaluation_code_version

    path = RESULTS / "variable_sv_evaluation.json"
    if not path.exists():
        pytest.skip("variable_sv_evaluation.json not generated yet")
    recorded = json.loads(path.read_text())["code_version"]
    current = evaluation_code_version(extra_paths=[
        vs.__file__, SCRIPTS / "run_variable_sv_evaluation.py",
        RESULTS / "common_sv_evaluation.json",
        SCRIPTS / "run_common_sv_evaluation.py",
        SCRIPTS / "run_coverage_evaluation.py",
        SCRIPTS / "run_covid_volatility_evaluation.py"])
    assert recorded == current, "re-run run_variable_sv_evaluation.py"

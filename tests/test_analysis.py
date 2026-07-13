"""Unit tests for boe_var.analysis on a tiny hand-built VAR(1)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from boe_var import analysis


class FakeDraw:
    """Minimal stand-in for a BVAR posterior draw (VAR(1), k variables)."""

    def __init__(self, A, Sigma):
        self.A = np.asarray(A, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        k = self.A.shape[0]
        # Pi = [A | const] with zero constant, layout k x (k*p + 1)
        self.Pi = np.hstack([self.A, np.zeros((k, 1))])

    def companion(self):
        return self.A  # VAR(1): companion is A itself


A = np.diag([0.5, 0.8])
B_ID = np.eye(2)
DRAW = FakeDraw(A, B_ID @ B_ID.T)


def test_irf_matches_companion_powers():
    H = 6
    out = analysis.irf(DRAW, B_ID, horizons=H)
    assert out.shape == (2, 2, H)
    for h in range(H):
        expected = np.linalg.matrix_power(A, h) @ B_ID
        np.testing.assert_allclose(out[:, :, h], expected, atol=1e-12)
    # diagonal companion + identity B: responses are 0.5**h and 0.8**h
    np.testing.assert_allclose(out[0, 0], 0.5 ** np.arange(H))
    np.testing.assert_allclose(out[1, 1], 0.8 ** np.arange(H))
    assert np.all(out[0, 1] == 0) and np.all(out[1, 0] == 0)


def test_irf_nontrivial_B():
    B = np.array([[1.0, 0.0], [0.5, 2.0]])
    out = analysis.irf(FakeDraw(A, B @ B.T), B, horizons=4)
    for h in range(4):
        np.testing.assert_allclose(
            out[:, :, h], np.linalg.matrix_power(A, h) @ B, atol=1e-12)


def test_fevd_sums_to_one_and_analytic_values():
    B = np.array([[1.0, 0.0], [0.5, 2.0]])
    draw = FakeDraw(A, B @ B.T)
    H = 8
    shares = analysis.fevd(draw, B, horizons=H)
    assert shares.shape == (2, 2, H)
    np.testing.assert_allclose(shares.sum(axis=1), np.ones((2, H)), atol=1e-12)
    # variable 0 loads only on shock 0 (B[0,1]=0, A diagonal)
    np.testing.assert_allclose(shares[0, 0], np.ones(H))
    np.testing.assert_allclose(shares[0, 1], np.zeros(H), atol=1e-12)
    # variable 1 on impact: 0.25 / (0.25 + 4)
    np.testing.assert_allclose(shares[1, 0, 0], 0.25 / 4.25)
    np.testing.assert_allclose(shares[1, 1, 0], 4.0 / 4.25)


def test_estimated_shocks_invert_B():
    B = np.array([[2.0, 0.0], [1.0, 1.0]])
    rng = np.random.default_rng(0)
    eps = rng.standard_normal((10, 2))
    u = eps @ B.T
    rec = analysis.estimated_shocks(FakeDraw(A, B @ B.T), B, u)
    np.testing.assert_allclose(rec, eps, atol=1e-12)


def test_historical_decomposition_sums_to_stochastic_component():
    B = np.array([[1.0, 0.0], [0.5, 2.0]])
    draw = FakeDraw(A, B @ B.T)
    rng = np.random.default_rng(1)
    T = 20
    u = rng.standard_normal((T, 2))
    # simulate y_t = A y_{t-1} + u_t from zero initial condition, plus a
    # deterministic drift so that "data minus deterministic" is the MA part
    y_stoch = np.zeros((T, 2))
    for t in range(T):
        prev = y_stoch[t - 1] if t > 0 else np.zeros(2)
        y_stoch[t] = A @ prev + u[t]
    det = np.outer(np.arange(T), [0.1, -0.2])
    y = y_stoch + det

    hd = analysis.historical_decomposition(draw, B, u)
    # components sum to the stochastic part = data minus deterministic part
    np.testing.assert_allclose(hd["shocks"].sum(axis=2), y - det, atol=1e-10)
    np.testing.assert_allclose(hd["stochastic"], y_stoch, atol=1e-10)
    np.testing.assert_allclose(hd["eps"],
                               np.linalg.solve(B, u.T).T, atol=1e-12)


def test_historical_decomposition_single_shock_columns():
    # With identity B and diagonal A, shock j contributes only to variable j.
    hd = analysis.historical_decomposition(
        DRAW, B_ID, np.random.default_rng(2).standard_normal((15, 2)))
    np.testing.assert_allclose(hd["shocks"][:, 0, 1], 0, atol=1e-12)
    np.testing.assert_allclose(hd["shocks"][:, 1, 0], 0, atol=1e-12)


def test_aggregate_median_and_bands():
    arrays = [np.full((3, 2), v) for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
    agg = analysis.aggregate(arrays)
    np.testing.assert_allclose(agg["median"], 3.0)
    assert np.all(agg["lo90"] <= agg["lo68"])
    assert np.all(agg["lo68"] <= agg["median"])
    assert np.all(agg["median"] <= agg["hi68"])
    assert np.all(agg["hi68"] <= agg["hi90"])


def test_weighted_quantile_uniform_matches_percentile():
    rng = np.random.default_rng(7)
    v = rng.standard_normal((9, 4, 3))
    w = np.ones(9)
    for q in [0.05, 0.16, 0.5, 0.84, 0.95]:
        np.testing.assert_allclose(
            analysis.weighted_quantile(v, q, w),
            np.percentile(v, 100 * q, axis=0), atol=1e-12)
    # scaling weights uniformly must not change anything
    np.testing.assert_allclose(
        analysis.weighted_quantile(v, 0.5, 3.7 * w),
        np.median(v, axis=0), atol=1e-12)


def test_weighted_quantile_extreme_weight_concentrates():
    v = np.arange(10.0)
    w = np.ones(10)
    w[7] = 1e6
    med = analysis.weighted_quantile(v, 0.5, w)
    np.testing.assert_allclose(med, 7.0, atol=1e-3)
    # quantiles just around the median stay pinned near the heavy value
    assert abs(analysis.weighted_quantile(v, 0.499, w) - 7.0) < 0.01
    assert abs(analysis.weighted_quantile(v, 0.501, w) - 7.0) < 0.01


def test_weighted_quantile_hand_computed_three_points():
    # values [1, 2, 4], weights [1, 2, 1], total W = 4
    # cumulative midpoints: c = [0.5, 2.0, 3.5]; rescaled to [0, 1]:
    # p = [0, 0.5, 1]. So q=0.25 interpolates halfway between 1 and 2 -> 1.5,
    # q=0.5 -> 2, q=0.75 -> halfway between 2 and 4 -> 3.
    v = np.array([1.0, 2.0, 4.0])
    w = np.array([1.0, 2.0, 1.0])
    np.testing.assert_allclose(analysis.weighted_quantile(v, 0.25, w), 1.5)
    np.testing.assert_allclose(analysis.weighted_quantile(v, 0.5, w), 2.0)
    np.testing.assert_allclose(analysis.weighted_quantile(v, 0.75, w), 3.0)
    # order-invariance
    perm = [2, 0, 1]
    np.testing.assert_allclose(
        analysis.weighted_quantile(v[perm], 0.75, w[perm]), 3.0)


def test_weighted_quantile_rejects_bad_weights():
    v = np.arange(3.0)
    with pytest.raises(ValueError):
        analysis.weighted_quantile(v, 0.5, np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        analysis.weighted_quantile(v, 0.5, np.zeros(3))
    with pytest.raises(ValueError):
        analysis.weighted_quantile(v, 0.5, np.ones(4))


def test_aggregate_with_weights_band_ordering_and_backcompat():
    rng = np.random.default_rng(11)
    arrays = [rng.standard_normal((3, 2)) for _ in range(25)]
    w = rng.uniform(0.1, 2.0, size=25)
    agg = analysis.aggregate(arrays, weights=w)
    assert np.all(agg["lo90"] <= agg["lo68"])
    assert np.all(agg["lo68"] <= agg["median"])
    assert np.all(agg["median"] <= agg["hi68"])
    assert np.all(agg["hi68"] <= agg["hi90"])
    # uniform weights reproduce the unweighted aggregate exactly
    agg_u = analysis.aggregate(arrays, weights=np.ones(25))
    agg_none = analysis.aggregate(arrays)
    for key in agg_none:
        np.testing.assert_allclose(agg_u[key], agg_none[key], atol=1e-12)


def test_band_helpers_accept_weights():
    B = np.array([[1.0, 0.0], [0.5, 2.0]])
    draws = [FakeDraw(A * s, B @ B.T) for s in (0.8, 0.9, 1.0)]
    pairs = [(d, B) for d in draws]
    w = np.array([0.2, 0.5, 0.3])
    ib = analysis.irf_bands(pairs, 6, weights=w)
    assert ib["median"].shape == (2, 2, 6)
    assert np.all(ib["lo68"] <= ib["hi68"])
    fb = analysis.fevd_bands(pairs, 6, weights=w)
    assert np.all(fb["lo90"] <= fb["hi90"])
    u = np.random.default_rng(4).standard_normal((10, 2))
    sb = analysis.shock_bands(pairs, lambda d: u, weights=w)
    assert sb["median"].shape == (10, 2)
    hd = analysis.hd_median(pairs, lambda d: u, weights=w)
    assert hd.shape == (10, 2, 2)


def test_plot_helpers_write_pngs(tmp_path):
    pytest.importorskip("matplotlib")
    B = np.array([[1.0, 0.0], [0.5, 2.0]])
    draw = FakeDraw(A, B @ B.T)
    pairs = [(draw, B)] * 3
    # patch names for k=2
    orig_v, orig_s = analysis.VARIABLE_NAMES, analysis.SHOCK_NAMES
    analysis.VARIABLE_NAMES = ["v1", "v2"]
    analysis.SHOCK_NAMES = ["s1", "s2"]
    try:
        ib = analysis.irf_bands(pairs, 8)
        p1 = analysis.plot_irf_grid(ib, [0, 1], str(tmp_path / "irf.png"))
        fb = analysis.fevd_bands(pairs, 8)
        p2 = analysis.plot_fevd(fb, str(tmp_path / "fevd.png"))
        u = np.random.default_rng(3).standard_normal((12, 2))
        sb = analysis.shock_bands(pairs, lambda d: u)
        p3 = analysis.plot_shocks(sb, str(tmp_path / "shocks.png"))
        hd = analysis.historical_decomposition(draw, B, u)
        p4 = analysis.plot_hist_decomp(
            hd["shocks"], np.zeros((12, 2)), [0, 1],
            str(tmp_path / "hd.png"))
        for p in (p1, p2, p3, p4):
            assert os.path.getsize(p) > 0
    finally:
        analysis.VARIABLE_NAMES, analysis.SHOCK_NAMES = orig_v, orig_s

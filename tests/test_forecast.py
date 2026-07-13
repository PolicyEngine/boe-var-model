"""Tests for boe_var.forecast (forecasting + Section 5 revision tool)."""

import numpy as np
import pytest

from boe_var.bvar import PosteriorDraw
from boe_var.forecast import (
    composite_irf,
    forecast_bands,
    forecast_revision,
    reduced_form_residuals,
    sample_forecast,
    shock_distribution_T,
    unconditional_forecast,
    yoy,
    yoy_transform_irf,
)


def make_draw(A_blocks, c, Sigma, n_dummies=0):
    """Build a PosteriorDraw from lag matrices A_blocks (list of k x k), const c."""
    k = A_blocks[0].shape[0]
    p = len(A_blocks)
    D = np.zeros((k, n_dummies))
    Pi = np.hstack(A_blocks + [np.asarray(c, float).reshape(k, 1), D])
    return PosteriorDraw(Pi=Pi, Sigma=np.asarray(Sigma, float), lags=p, k=k)


RNG = np.random.default_rng(0)


def stable_var(k=2, p=1, rng=RNG, n_dummies=0):
    A = [0.5 * rng.standard_normal((k, k)) / (l + 1) for l in range(p)]
    # scale to be comfortably stable
    for i in range(p):
        A[i] *= 0.5
    c = rng.standard_normal(k)
    S = rng.standard_normal((k, k))
    Sigma = S @ S.T + 0.5 * np.eye(k)
    return make_draw(A, c, Sigma, n_dummies=n_dummies)


# ---------------------------------------------------------------------------
def test_var1_deterministic_matches_analytic():
    """Deterministic forecast of a VAR(1) equals the analytic A^h iteration."""
    A = np.array([[0.5, 0.1], [-0.2, 0.7]])
    c = np.array([1.0, -0.5])
    draw = make_draw([A], c, np.eye(2))
    y_hist = np.array([[0.3, 0.4], [1.0, 2.0]])  # last row is the state
    H = 10
    fc = unconditional_forecast(draw, y_hist, horizons=H)
    y = y_hist[-1]
    for h in range(H):
        y = A @ y + c
        np.testing.assert_allclose(fc[h], y, rtol=0, atol=1e-12)


def test_var2_uses_two_lags():
    draw = stable_var(k=2, p=2)
    A1 = draw.Pi[:, :2]
    A2 = draw.Pi[:, 2:4]
    c = draw.Pi[:, 4]
    y_hist = RNG.standard_normal((5, 2))
    fc = unconditional_forecast(draw, y_hist, horizons=3)
    y1 = A1 @ y_hist[-1] + A2 @ y_hist[-2] + c
    y2 = A1 @ y1 + A2 @ y_hist[-1] + c
    y3 = A1 @ y2 + A2 @ y1 + c
    np.testing.assert_allclose(fc, np.vstack([y1, y2, y3]), atol=1e-12)


def test_yoy():
    x = np.arange(10.0)
    np.testing.assert_allclose(yoy(x), np.full(6, 4.0))
    # 2-d: per column
    X = np.column_stack([np.arange(8.0), np.arange(8.0) ** 2])
    out = yoy(X)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[:, 0], 4.0)
    np.testing.assert_allclose(out[:, 1], X[4:, 1] - X[:-4, 1])
    with pytest.raises(ValueError):
        yoy(np.arange(4.0))


def test_sample_forecast_reduces_to_deterministic_when_sigma_zero():
    draw = stable_var(k=3, p=2)
    draw0 = PosteriorDraw(Pi=draw.Pi, Sigma=np.zeros((3, 3)),
                          lags=draw.lags, k=draw.k)
    y_hist = RNG.standard_normal((6, 3))
    det = unconditional_forecast(draw0, y_hist, horizons=8)
    sto = sample_forecast(draw0, y_hist, horizons=8,
                          rng=np.random.default_rng(1))
    np.testing.assert_allclose(sto, det, atol=1e-10)


def test_forecast_bands_ordering_weighted():
    draws = [stable_var(k=2, p=1, rng=np.random.default_rng(i)) for i in range(6)]
    y_hist = np.zeros((3, 2))
    w = np.random.default_rng(2).uniform(0.1, 2.0, size=6)
    b = forecast_bands(draws, y_hist, horizons=6, weights=w, n_paths=4,
                       rng=np.random.default_rng(3))
    assert np.all(b["lo90"] <= b["lo68"] + 1e-12)
    assert np.all(b["lo68"] <= b["median"] + 1e-12)
    assert np.all(b["median"] <= b["hi68"] + 1e-12)
    assert np.all(b["hi68"] <= b["hi90"] + 1e-12)
    # also accepts (draw, B, w) triples
    triples = [(d, np.linalg.cholesky(d.Sigma), wi) for d, wi in zip(draws, w)]
    b2 = forecast_bands(triples, y_hist, horizons=6, weights=w, n_paths=2,
                        rng=np.random.default_rng(4))
    assert b2["median"].shape == (6, 2)


def _simulate(draw, T, rng):
    k, p = draw.k, draw.lags
    A = [draw.Pi[:, l * k:(l + 1) * k] for l in range(p)]
    c = draw.Pi[:, k * p]
    L = np.linalg.cholesky(draw.Sigma)
    y = np.zeros((T, k))
    for t in range(p, T):
        y[t] = c + sum(A[l] @ y[t - 1 - l] for l in range(p)) \
            + L @ rng.standard_normal(k)
    return y


def test_adding_up_identity_per_draw():
    """forecast_T = pseudo_{T-1} + composite IRF, exact per draw (1e-8)."""
    rng = np.random.default_rng(7)
    for p in (1, 3):
        draw = stable_var(k=3, p=p, rng=rng)
        B = np.linalg.cholesky(draw.Sigma)  # any valid factorization
        y = _simulate(draw, 40, rng)
        res = forecast_revision(draw, B, y, horizons=9, atol=1e-8)
        assert res["max_err"] <= 1e-8
        assert res["fc_T"].shape == (9, 3)
        assert res["fc_Tm1"].shape == (10, 3)
        assert res["composite"].shape == (10, 3)
        # explicit spot check at h = 1
        np.testing.assert_allclose(
            res["fc_T"][0], res["fc_Tm1"][1] + res["composite"][1], atol=1e-8)
        # eps_T consistent with reduced-form residual at T
        u = reduced_form_residuals(draw, y)
        np.testing.assert_allclose(B @ res["eps_T"], u[-1], atol=1e-8)


def test_shock_distribution_and_composite():
    rng = np.random.default_rng(11)
    draw = stable_var(k=2, p=1, rng=rng)
    B = np.linalg.cholesky(draw.Sigma)
    y = _simulate(draw, 30, rng)
    pairs = [(draw, B), (draw, -B)]  # sign-flipped twin
    dist = shock_distribution_T(pairs, lambda d: reduced_form_residuals(d, y))
    assert dist["samples"].shape == (2, 2)
    np.testing.assert_allclose(dist["samples"][0], -dist["samples"][1])
    assert np.all((dist["p_pos"] >= 0) & (dist["p_pos"] <= 1))

    out = composite_irf(pairs, weights=np.array([1.0, 1.0]),
                        eps_T=dist["samples"], horizons=6)
    # joint composite is invariant to the B sign flip: identical across pairs
    med = out["joint"]["median"]
    assert med.shape == (6, 2)
    np.testing.assert_allclose(out["joint"]["lo90"], out["joint"]["hi90"],
                               atol=1e-10)
    # decomposition sums to the joint effect
    np.testing.assert_allclose(out["decomp_median"].sum(axis=2), med,
                               atol=1e-10)


def test_yoy_transform_irf():
    x = np.arange(9.0)[:, None]  # level effect grows linearly
    out = yoy_transform_irf(x)
    np.testing.assert_allclose(out[:4], x[:4])   # h<4: level itself
    np.testing.assert_allclose(out[4:], 4.0)     # thereafter x[h]-x[h-4]=4

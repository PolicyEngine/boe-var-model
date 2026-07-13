import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boe_var.bvar import BVAR, PosteriorDraw


def simulate_var1(T=500, seed=0, dummies=None, dummy_coefs=None):
    """Simulate a stationary VAR(1), k=2, with known coefficients."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.6, 0.15], [0.1, 0.7]])
    c = np.array([0.5, -0.3])
    L = np.array([[1.0, 0.0], [0.4, 0.8]])
    y = np.zeros((T + 100, 2))
    for t in range(1, T + 100):
        y[t] = c + A @ y[t - 1] + L @ rng.standard_normal(2)
    y = y[100:]
    if dummies is not None:
        y = y + dummies @ dummy_coefs.T
    return y, A, c, L @ L.T


def test_posterior_mean_close_to_truth():
    y, A, c, Sig = simulate_var1()
    model = BVAR(y, lags=1, lam=0.5)
    draws = model.sample_posterior(500, seed=42)
    Pi_mean = np.mean([d.Pi for d in draws], axis=0)
    A_hat = Pi_mean[:, :2]
    c_hat = Pi_mean[:, 2]
    Sig_mean = np.mean([d.Sigma for d in draws], axis=0)
    assert np.allclose(A_hat, A, atol=0.1)
    assert np.allclose(c_hat, c, atol=0.25)
    assert np.allclose(Sig_mean, Sig, atol=0.25)


def test_shapes_and_companion():
    y, *_ = simulate_var1(T=200)
    model = BVAR(y, lags=4)
    draws = model.sample_posterior(3, seed=1)
    assert len(draws) == 3
    k, p = 2, 4
    m = k * p + 1
    for d in draws:
        assert isinstance(d, PosteriorDraw)
        assert d.Pi.shape == (k, m)
        assert d.Sigma.shape == (k, k)
        # Sigma symmetric PD
        assert np.allclose(d.Sigma, d.Sigma.T)
        assert np.all(np.linalg.eigvalsh(d.Sigma) > 0)
        F = d.companion()
        assert F.shape == (k * p, k * p)
        assert np.allclose(F[:k, :], d.Pi[:, : k * p])
        assert np.allclose(F[k:, : k * (p - 1)], np.eye(k * (p - 1)))
    u = model.residuals(draws[0])
    assert u.shape == (200 - p, k)
    # residuals consistent with layout: Y - X Pi'
    assert np.allclose(u, model.Y - model.X @ draws[0].Pi.T)


def test_dummies_handled():
    T = 500
    dummies = np.zeros((T, 2))
    dummies[100:110, 0] = 1.0
    dummies[300:305, 1] = 1.0
    dummy_coefs = np.array([[5.0, 0.0], [0.0, -4.0]])
    y, *_ = simulate_var1(T=T, seed=3, dummies=dummies, dummy_coefs=dummy_coefs)
    model = BVAR(y, lags=1, dummies=dummies)
    assert model.m == 2 * 1 + 1 + 2
    draws = model.sample_posterior(200, seed=7)
    Pi_mean = np.mean([d.Pi for d in draws], axis=0)
    assert Pi_mean.shape == (2, 5)
    D_hat = Pi_mean[:, 3:]  # dummy block after lags + constant
    # signs of the additive shifts should be recovered
    assert D_hat[0, 0] > 1.0
    assert D_hat[1, 1] < -1.0


def test_reproducibility_with_seed():
    y, *_ = simulate_var1(T=300, seed=5)
    model = BVAR(y, lags=2)
    d1 = model.sample_posterior(5, seed=123)
    d2 = model.sample_posterior(5, seed=123)
    d3 = model.sample_posterior(5, seed=124)
    for a, b in zip(d1, d2):
        assert np.array_equal(a.Pi, b.Pi)
        assert np.array_equal(a.Sigma, b.Sigma)
    assert not np.allclose(d1[0].Pi, d3[0].Pi)


def test_input_validation():
    y, *_ = simulate_var1(T=100)
    with pytest.raises(ValueError):
        BVAR(y[:, 0])  # 1-D
    with pytest.raises(ValueError):
        BVAR(y, lags=1, dummies=np.zeros((50, 1)))  # wrong length

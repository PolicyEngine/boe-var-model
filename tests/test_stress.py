"""Numerical-reliability / stress tests.

These exercise the estimator and identification under a fixed seed, edge-case
restriction sets, minimal draws and a short sample, and assert the numerical
guarantees the pipeline depends on (Sigma stays PSD, no blow-ups, VAR roots
near the unit circle, clean rejection of impossible restrictions,
determinism).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import boe_var.identification as ident  # noqa: E402
from boe_var.bvar import BVAR  # noqa: E402
from boe_var.identification import (  # noqa: E402
    K,
    SHOCKS,
    SIGN_RESTRICTIONS,
    ZERO_RESTRICTIONS,
    draw_Q,
    identify,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20240715)


def _restriction_consistent_sigma(rng, jitter=0.0):
    """Sigma = B0 B0' with B0 satisfying every Table 2 zero/sign restriction,
    so the sign-acceptance region carries real probability mass. ``jitter``
    perturbs B0 to make a family of distinct-but-acceptable covariances.

    Identification is hard-wired to the 8-variable BoE system (K = 8), so all
    identify()-level tests must feed it K x K covariances."""
    B0 = 0.05 * rng.standard_normal((K, K))
    for j, shock in enumerate(SHOCKS):
        for v, s in SIGN_RESTRICTIONS[shock]:
            B0[v, j] = s * 1.0
        for v in ZERO_RESTRICTIONS[shock]:
            B0[v, j] = 0.0
    B0 = B0 + jitter * rng.standard_normal((K, K))
    return B0 @ B0.T


def _stable_var_data(T=160, k=3, seed=0):
    """Simulate a comfortably stationary VAR(1) for reuse across tests."""
    rng = np.random.default_rng(seed)
    A = 0.5 * np.eye(k) + 0.05 * rng.standard_normal((k, k))
    L = np.tril(0.3 * rng.standard_normal((k, k))) + np.eye(k)
    y = np.zeros((T, k))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + L @ rng.standard_normal(k)
    return y


# ---------------------------------------------------------------------------
# Posterior / covariance stability
# ---------------------------------------------------------------------------

def test_sampled_sigma_is_symmetric_and_psd():
    """Every posterior Sigma draw is symmetric positive definite and finite."""
    y = _stable_var_data(T=140, k=3, seed=1)
    model = BVAR(y, lags=2)
    for d in model.sample_posterior(80, seed=7):
        assert np.all(np.isfinite(d.Sigma))
        assert np.allclose(d.Sigma, d.Sigma.T, atol=1e-10)
        eig = np.linalg.eigvalsh(d.Sigma)
        assert eig.min() > 0, "Sigma not PD"
        # Cholesky must succeed (identification relies on it)
        np.linalg.cholesky(d.Sigma)


def test_posterior_draws_do_not_blow_up():
    """Coefficients and covariances stay finite and bounded across many
    draws (no numerical blow-up in the NIW sampler)."""
    y = _stable_var_data(T=160, k=3, seed=2)
    model = BVAR(y, lags=2)
    draws = model.sample_posterior(120, seed=11)
    for d in draws:
        assert np.all(np.isfinite(d.Pi))
        assert np.all(np.isfinite(d.Sigma))
        assert np.linalg.cond(d.Sigma) < 1e10


def test_var_companion_eigenvalues_near_unit_circle():
    """The estimated real-data system is (near-)stable: companion eigenvalues
    lie inside or only marginally outside the unit circle. The eight variables
    are (near-unit-root) 100*log levels, so the dominant root sits close to 1
    -- but a wildly explosive root (|lambda| >> 1) would signal a broken
    estimate."""
    import pandas as pd

    from boe_var.data import load_data

    df = load_data()
    df = df.loc[(df.index >= pd.Period("1992Q1", "Q"))
                & (df.index <= pd.Period("2023Q2", "Q"))]
    y = df.to_numpy(dtype=float)
    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    dummies = np.column_stack([(df.index == q).astype(float) for q in quarters])
    model = BVAR(y, lags=4, dummies=dummies, lam=0.2, mu=1.0, theta=1.0)
    for d in model.sample_posterior(60, seed=3):
        rho = np.abs(np.linalg.eigvals(d.companion())).max()
        assert rho < 1.10, f"explosive companion spectral radius {rho}"


# ---------------------------------------------------------------------------
# Identification edge cases
# ---------------------------------------------------------------------------

def test_impossible_sign_restrictions_reject_cleanly(monkeypatch, rng):
    """A contradictory sign restriction must make identify() return an empty
    list -- never a draw that silently violates the restriction."""
    from types import SimpleNamespace

    bad = {s: list(r) for s, r in ident.SIGN_RESTRICTIONS.items()}
    # world_demand requires UK GDP (row 7) to be both > 0 and < 0.
    bad["world_demand"] = [(7, +1), (7, -1)]
    monkeypatch.setattr(ident, "SIGN_RESTRICTIONS", bad)

    def spd():
        A = rng.standard_normal((K, K))
        return A @ A.T + K * np.eye(K)

    draws = [SimpleNamespace(Sigma=spd(), Pi=None) for _ in range(50)]
    out = identify(draws, rng=rng, compute_weights=False)
    assert out == []
    assert identify.last_acceptance_rate == 0.0


def test_draw_Q_raises_when_zero_restrictions_are_infeasible(rng):
    """More zero restrictions than dimensions on one shock leaves no feasible
    direction -> draw_Q raises rather than returning a bad column."""
    A = rng.standard_normal((K, K))
    Sigma = A @ A.T + K * np.eye(K)
    # give the first shock a zero on every variable (K constraints in R^K):
    # the very first column processed then has an empty null space.
    infeasible = {s: [] for s in SHOCKS}
    infeasible[SHOCKS[0]] = list(range(K))
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        draw_Q(Sigma, rng, zero_restrictions=infeasible)


def test_draw_Q_zero_restrictions_hold_and_Q_orthogonal(rng):
    """draw_Q output is orthogonal and enforces the real zero block exactly."""
    A = rng.standard_normal((K, K))
    Sigma = A @ A.T + K * np.eye(K)
    for _ in range(30):
        Q = draw_Q(Sigma, rng)
        assert np.allclose(Q @ Q.T, np.eye(K), atol=1e-8)
        B = np.linalg.cholesky(Sigma) @ Q
        for j, shock in enumerate(SHOCKS):
            for v in ident.ZERO_RESTRICTIONS[shock]:
                assert abs(B[v, j]) < 1e-10


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_identify_is_deterministic_under_fixed_seed(rng):
    """Same posterior draws + same RNG seed -> identical accepted impact
    matrices (byte-for-byte), so CI results are reproducible."""
    from types import SimpleNamespace

    draws = [SimpleNamespace(Sigma=_restriction_consistent_sigma(rng, 0.02),
                             Pi=None)
             for _ in range(40)]
    a = identify(draws, rng=np.random.default_rng(99), compute_weights=False)
    b = identify(draws, rng=np.random.default_rng(99), compute_weights=False)
    assert len(a) == len(b) and len(a) > 0
    for (da, Ba, _), (db, Bb, _) in zip(a, b):
        assert da is db
        np.testing.assert_array_equal(Ba, Bb)


# ---------------------------------------------------------------------------
# Minimal draws / short sample
# ---------------------------------------------------------------------------

def test_single_posterior_draw_identifies_or_rejects_cleanly(rng):
    """A one-draw run must not crash: it returns either an empty list or a
    single valid, restriction-satisfying triple."""
    from types import SimpleNamespace

    draws = [SimpleNamespace(Sigma=_restriction_consistent_sigma(rng), Pi=None)]
    out = identify(draws, rng=np.random.default_rng(1), compute_weights=False)
    assert isinstance(out, list) and len(out) <= 1
    for draw, B, w in out:
        assert np.all(np.isfinite(B))
        assert np.allclose(B @ B.T, draw.Sigma, atol=1e-7)
        assert np.isfinite(w) and w > 0


def test_short_sample_estimates_without_nan():
    """A short estimation sample (barely enough observations) still yields a
    finite, PSD posterior and no NaN/inf."""
    y = _stable_var_data(T=40, k=3, seed=6)
    model = BVAR(y, lags=2)
    draws = model.sample_posterior(20, seed=31)
    assert len(draws) == 20
    for d in draws:
        assert np.all(np.isfinite(d.Pi))
        assert np.all(np.isfinite(d.Sigma))
        assert np.linalg.eigvalsh(d.Sigma).min() > 0

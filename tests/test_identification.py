import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boe_var.identification import (  # noqa: E402
    K,
    SHOCKS,
    SIGN_RESTRICTIONS,
    ZERO_RESTRICTIONS,
    check_signs,
    draw_B,
    draw_Q,
    identify,
    structural_shocks,
)


def random_spd(rng, k=K):
    A = rng.standard_normal((k, k))
    return A @ A.T + k * np.eye(k)


def restriction_consistent_sigma(rng):
    """Sigma = B0 B0' with B0 satisfying all zero/sign restrictions, so the
    sign-acceptance region has non-trivial probability mass."""
    B0 = 0.05 * rng.standard_normal((K, K))
    for j, shock in enumerate(SHOCKS):
        for v, s in SIGN_RESTRICTIONS[shock]:
            B0[v, j] = s * 1.0
        for v in ZERO_RESTRICTIONS[shock]:
            B0[v, j] = 0.0
    return B0 @ B0.T


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def test_B_factorizes_sigma_and_Q_orthogonal(rng):
    Sigma = random_spd(rng)
    Q = draw_Q(Sigma, rng)
    assert np.allclose(Q @ Q.T, np.eye(K), atol=1e-8)
    assert np.allclose(Q.T @ Q, np.eye(K), atol=1e-8)
    B = np.linalg.cholesky(Sigma) @ Q
    assert np.allclose(B @ B.T, Sigma, atol=1e-8)


def test_zero_restrictions_hold(rng):
    Sigma = random_spd(rng)
    for _ in range(20):
        Q = draw_Q(Sigma, rng)
        B = np.linalg.cholesky(Sigma) @ Q
        for j, shock in enumerate(SHOCKS):
            for v in ZERO_RESTRICTIONS[shock]:
                assert abs(B[v, j]) < 1e-10, (shock, v, B[v, j])


def test_accepted_draws_satisfy_signs(rng):
    Sigma = restriction_consistent_sigma(rng)
    accepted = 0
    for _ in range(20000):
        B = draw_B(Sigma, rng)
        if B is None:
            continue
        accepted += 1
        for j, shock in enumerate(SHOCKS):
            for v, s in SIGN_RESTRICTIONS[shock]:
                assert s * B[v, j] > 0
        # zeros still hold after normalization flips
        for j, shock in enumerate(SHOCKS):
            for v in ZERO_RESTRICTIONS[shock]:
                assert abs(B[v, j]) < 1e-10
        if accepted >= 3:
            break
    assert accepted >= 1


def test_infeasible_restriction_never_accepts(rng):
    # Contradictory: uk_gdp response to world_demand both > 0 and < 0.
    infeasible = {s: list(r) for s, r in SIGN_RESTRICTIONS.items()}
    infeasible["world_demand"] = [(7, +1), (7, -1)]
    Sigma = random_spd(rng)
    L = np.linalg.cholesky(Sigma)
    for _ in range(200):
        B = L @ draw_Q(Sigma, rng)
        assert check_signs(B, infeasible) is None


def test_identify_returns_pairs(rng):
    draws = [
        SimpleNamespace(Sigma=restriction_consistent_sigma(rng)) for _ in range(5)
    ]
    out = identify(draws, target_accepted=3, max_tries_per_draw=3000, rng=rng)
    assert 0 < len(out) <= 3
    for draw, B in out:
        assert np.allclose(B @ B.T, draw.Sigma, atol=1e-8)
        assert check_signs(B) is not None


def test_structural_shocks_roundtrip(rng):
    Sigma = random_spd(rng)
    B = np.linalg.cholesky(Sigma) @ draw_Q(Sigma, rng)
    eps = rng.standard_normal((30, K))
    u = eps @ B.T
    assert np.allclose(structural_shocks(B, u), eps, atol=1e-8)

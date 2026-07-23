import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import boe_var.identification as ident  # noqa: E402
from boe_var.identification import (  # noqa: E402
    BLOCKS,
    K,
    SHOCKS,
    SIGN_RESTRICTIONS,
    ZERO_RESTRICTIONS,
    check_signs,
    draw_B,
    draw_Q,
    ess,
    identify,
    importance_weight,
    log_importance_weight,
    permutation_search,
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


def test_identify_returns_triples(rng):
    Sigma = restriction_consistent_sigma(rng)
    draws = [SimpleNamespace(Sigma=Sigma) for _ in range(400)]
    out = identify(draws, rng=rng, target_accepted=3)
    assert 0 < len(out) <= 3
    for draw, B, w in out:
        assert np.allclose(B @ B.T, draw.Sigma, atol=1e-8)
        assert check_signs(B) is not None
        assert np.isfinite(w) and w > 0
    assert 0.0 < identify.last_acceptance_rate <= 1.0


# ---------------------------------------------------------------------------
# Chan-Matthes-Yu blockwise permutation search
# ---------------------------------------------------------------------------


def satisfying_B(rng):
    """A B that satisfies all zero and sign restrictions exactly."""
    B = 0.05 * rng.standard_normal((K, K))
    for j, shock in enumerate(SHOCKS):
        for v, s in SIGN_RESTRICTIONS[shock]:
            B[v, j] = s * 1.0
        for v in ZERO_RESTRICTIONS[shock]:
            B[v, j] = 0.0
    return B


def shuffle_within_blocks(B, rng):
    out = B.copy()
    for block in BLOCKS:
        perm = rng.permutation(len(block))
        for a, j in enumerate(block):
            out[:, j] = rng.choice([-1.0, 1.0]) * B[:, block[perm[a]]]
    return out


def test_permutation_search_recovers_shuffled_columns(rng):
    B0 = satisfying_B(rng)
    for _ in range(10):
        shuffled = shuffle_within_blocks(B0, rng)
        found = permutation_search(shuffled, rng)
        assert found is not None
        assert check_signs(found) is not None
        for j, shock in enumerate(SHOCKS):
            for v in ZERO_RESTRICTIONS[shock]:
                assert abs(found[v, j]) < 1e-10
            for v, s in SIGN_RESTRICTIONS[shock]:
                assert s * found[v, j] > 0


def test_permutation_search_rejects_infeasible(rng):
    B = satisfying_B(rng)
    # Make every UK-block column violate uk_demand ((3,+),(5,+),(7,+))
    # both as-is and negated: pattern (+, +, -) fails, negation (-,-,+) fails.
    for j in [4, 5, 6, 7]:
        B[3, j], B[5, j], B[7, j] = 1.0, 1.0, -1.0
    assert permutation_search(B, rng) is None


def test_identify_one_Q_per_draw(rng, monkeypatch):
    calls = {"n": 0}
    orig = ident.draw_Q

    def counting_draw_Q(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(ident, "draw_Q", counting_draw_Q)
    draws = [SimpleNamespace(Sigma=random_spd(rng)) for _ in range(25)]
    ident.identify(draws, rng=rng)
    assert calls["n"] == 25


# ---------------------------------------------------------------------------
# Arias et al. (2018) importance weights
# ---------------------------------------------------------------------------


def test_weight_positive_finite_and_ve_f_term(rng):
    Sigma = random_spd(rng)
    Q = draw_Q(Sigma, rng)
    n, m = K, 0
    log_w = log_importance_weight((np.zeros((0, K)), Sigma), Q)
    assert np.isfinite(log_w)
    w = importance_weight((np.zeros((0, K)), Sigma), Q)
    assert np.isfinite(w) and w > 0

    # log ve_f = -(2n+m+1) log|det A0|, computed independently
    L = np.linalg.cholesky(Sigma)
    A0 = np.linalg.solve(L.T, Q)
    log_ve_f = -(2 * n + m + 1) * np.log(abs(np.linalg.det(A0)))
    d = n * (n + m)
    x = np.concatenate([A0.reshape(-1, order="F"), np.zeros(0)])
    gf0 = ident._g_fh(x, n, m)
    n_zeros = ident._n_zero_restrictions()
    # dimension consistency: image dim (each w_j has one redundant sphere
    # coordinate) equals dim(x) - #zeros
    assert gf0.size - n == d - n_zeros
    Dz = ident._num_jacobian(lambda v: ident._zero_fn(v, n, m), x)
    Dgf = ident._num_jacobian(lambda v: ident._g_fh(v, n, m), x)
    DN = Dgf @ ident._null_qr(Dz, d)
    log_ve_gf = 0.5 * np.linalg.slogdet(DN.T @ DN)[1]
    assert np.isclose(log_w, log_ve_f - log_ve_gf, rtol=1e-8, atol=1e-8)


def test_weight_with_coefficients(rng):
    """Weight is finite/positive when a reduced-form coefficient block (m>0)
    is present, via a PosteriorDraw-like object."""
    Sigma = random_spd(rng)
    m = 3
    draw = SimpleNamespace(Sigma=Sigma, Pi=0.1 * rng.standard_normal((K, m)))
    Q = draw_Q(Sigma, rng)
    lw = log_importance_weight(draw, Q)
    assert np.isfinite(lw)


def test_weight_invariant_to_within_block_permutation(rng):
    Sigma = random_spd(rng)
    Q = draw_Q(Sigma, rng)
    coefs = (np.zeros((0, K)), Sigma)
    lw = log_importance_weight(coefs, Q)

    # permute block A columns {0,1,3} and flip a sign in the UK block
    Q2 = Q.copy()
    Q2[:, [0, 1, 3]] = Q[:, [3, 0, 1]]
    Q2[:, 5] = -Q[:, 5]
    lw2 = log_importance_weight(coefs, Q2)
    assert np.isclose(lw, lw2, rtol=1e-4, atol=1e-4)

    # permutation within the UK block {4,5,6,7}
    Q3 = Q.copy()
    Q3[:, [4, 5, 6, 7]] = Q[:, [6, 7, 4, 5]]
    lw3 = log_importance_weight(coefs, Q3)
    assert np.isclose(lw, lw3, rtol=1e-4, atol=1e-4)


def test_ess():
    assert np.isclose(ess([1.0, 1.0, 1.0, 1.0]), 4.0)
    assert np.isclose(ess([1.0, 0.0, 0.0]), 1.0)
    assert ess([]) == 0.0


def test_end_to_end_smoke_with_weights(rng):
    Sigma = restriction_consistent_sigma(rng)
    draws = [
        SimpleNamespace(Sigma=Sigma, Pi=0.05 * rng.standard_normal((K, 2)))
        for _ in range(200)
    ]
    out = identify(draws, rng=rng, target_accepted=2)
    assert len(out) >= 1
    ws = [w for _, _, w in out]
    assert all(np.isfinite(w) and w > 0 for w in ws)
    assert np.isclose(np.mean(ws), 1.0)  # normalized to mean 1
    n_eff = ess(ws)
    assert 0 < n_eff <= len(ws)
    for draw, B, _ in out:
        assert np.allclose(B @ B.T, draw.Sigma, atol=1e-8)
        assert check_signs(B) is not None


def test_structural_shocks_roundtrip(rng):
    Sigma = random_spd(rng)
    B = np.linalg.cholesky(Sigma) @ draw_Q(Sigma, rng)
    eps = rng.standard_normal((30, K))
    u = eps @ B.T
    assert np.allclose(structural_shocks(B, u), eps, atol=1e-8)


def test_identify_diagnostics_and_weak_inference_warnings(rng):
    from boe_var.bvar import PosteriorDraw
    from boe_var.identification import (
        ess, identify, weak_inference_warnings)

    # Diagnostics attached after a run (tiny run -> warnings expected).
    k = len(ident.VARIABLES)
    draws = []
    rng2 = np.random.default_rng(0)
    for _ in range(30):
        A = rng2.standard_normal((k, k + 4))
        Sigma = A @ A.T / (k + 4)
        draws.append(PosteriorDraw(Pi=np.zeros((k, k + 1)), Sigma=Sigma,
                                   lags=1, k=k))
    triples = identify(draws, rng=rng, compute_weights=False)
    diag = identify.last_diagnostics
    assert diag["attempted"] == 30
    assert diag["accepted"] == len(triples)
    if triples:
        w = np.array([t[2] for t in triples])
        assert diag["ess"] == pytest.approx(ess(w))
    assert isinstance(diag["warnings"], list)
    # 30 attempts can never satisfy the >=100 thresholds.
    assert diag["warnings"]

    # Threshold behavior of the helper itself.
    assert weak_inference_warnings(500, 500.0, 1000) == []
    msgs = weak_inference_warnings(24, 7.4, 300)
    assert len(msgs) == 2
    assert any("24 accepted" in m for m in msgs)
    assert any("ESS = 7.4" in m for m in msgs)
    assert all("draws >=" in m for m in msgs)
    # ESS-only warning.
    msgs = weak_inference_warnings(150, 40.0, 1000)
    assert len(msgs) == 1 and "ESS" in msgs[0]

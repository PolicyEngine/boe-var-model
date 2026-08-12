"""Cross-verification of the ARRW (2018) zero-restriction importance weight.

The weight's volume element is computed in ``identification.py`` through the
null space of the numerical Jacobian of the zero-restriction map (the
bsvarSIGNs route, with a smooth null-space basis). These tests verify it
end-to-end against an INDEPENDENT construction that never touches that
machinery: because the impact matrix satisfies B = chol(Sigma) @ Q, the zero
restrictions are literal zero ENTRIES of B, so the restricted manifold can be
parameterized explicitly by the free entries of B (plus Aplus), giving a
second, independent orthonormal tangent basis and volume element.

A further test documents why we deliberately deviate from bsvarSIGNs'
numerics: its C++ g_fh builds null-space bases with Armadillo's SVD-based
null(), which is discontinuous under the finite-difference steps used to
compute the Jacobian (measured jumps of order 1e6 * h), so its volume
element is dominated by basis-jump noise. Our _null_smooth basis is locally
smooth by construction; the test asserts our g.f map is finite-difference
continuous.
"""

from __future__ import annotations

import numpy as np
import pytest

from boe_var import identification as ident
from boe_var.identification import (
    K,
    SHOCKS,
    ZERO_RESTRICTIONS,
    _g_fh,
    _num_jacobian,
    draw_Q,
    log_importance_weight,
)

FREE_ENTRIES = [
    (v, j)
    for j in range(K)
    for v in range(K)
    if v not in ZERO_RESTRICTIONS[SHOCKS[j]]
]


def _independent_log_weight(B_coef: np.ndarray, Sigma: np.ndarray,
                            Q: np.ndarray) -> float:
    """ARRW log weight via explicit parameterization of the zero manifold.

    The manifold {x = [vec(A0); vec(Aplus)] : impact zeros hold} is the
    image of psi(theta) with theta = (free entries of B = chol(Sigma) Q,
    entries of Aplus): A0 = inv(B'), Aplus free. The volume element is
    computed on the orthonormalized columns of D(psi) -- no Dz, no null
    spaces. Volume elements are invariant to the orthonormal tangent basis,
    so this must agree with the package's null(Dz) construction exactly.
    """
    m = B_coef.shape[0]
    n_free = len(FREE_ENTRIES)

    def psi(theta):
        B = np.zeros((K, K))
        for t, (v, j) in zip(theta[:n_free], FREE_ENTRIES):
            B[v, j] = t
        A0 = np.linalg.inv(B.T)
        # x = [vec(A0); vec(Aplus)]; the Aplus coordinates are free and
        # pass through unchanged.
        return np.concatenate([A0.reshape(-1, order="F"),
                               theta[n_free:]])

    L = np.linalg.cholesky(Sigma)
    B = L @ Q
    A0 = np.linalg.solve(L.T, Q)
    Aplus = B_coef @ A0
    theta0 = np.concatenate([
        np.array([B[v, j] for v, j in FREE_ENTRIES]),
        Aplus.reshape(-1, order="F"),
    ])
    x = np.concatenate([A0.reshape(-1, order="F"),
                        Aplus.reshape(-1, order="F")])
    assert np.allclose(psi(theta0), x, atol=1e-9)

    Dpsi = _num_jacobian(psi, theta0)
    U, _ = np.linalg.qr(Dpsi)                    # orthonormal tangent basis
    Dgf = _num_jacobian(lambda v: _g_fh(v, K, m), x)
    DN = Dgf @ U
    # slogdet computes the raw determinant internally for the sign, which
    # under/overflows on this well-conditioned but large Gram matrix and
    # emits spurious warnings; the log-determinant it returns is correct.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        _, ld = np.linalg.slogdet(DN.T @ DN)
        _, ldA0 = np.linalg.slogdet(A0)
    return float(-(2 * K + m + 1) * ldA0 - 0.5 * ld)


def _random_case(rng, m=0):
    A = rng.standard_normal((K, K + 4))
    Sigma = A @ A.T / (K + 4)
    B_coef = 0.1 * rng.standard_normal((m, K))
    Q = draw_Q(Sigma, rng)
    return B_coef, Sigma, Q


@pytest.mark.parametrize("m", [0, 2])
def test_weight_matches_independent_parameterization(m):
    """Null-space route == explicit-parameterization route, full 8-var set."""
    rng = np.random.default_rng(7)
    for _ in range(2):
        B_coef, Sigma, Q = _random_case(rng, m=m)
        lw_pkg = log_importance_weight((B_coef, Sigma), Q)
        lw_ind = _independent_log_weight(B_coef, Sigma, Q)
        assert lw_pkg == pytest.approx(lw_ind, abs=1e-6)


def test_weight_invariant_to_reference_basis():
    """The volume element must not depend on the smooth-basis reference."""
    rng = np.random.default_rng(11)
    B_coef, Sigma, Q = _random_case(rng, m=1)
    lw1 = log_importance_weight((B_coef, Sigma), Q)
    saved = dict(ident._REF_CACHE)
    try:
        ident._REF_CACHE.clear()
        ident._REF_CACHE[K] = np.random.default_rng(999).standard_normal(
            (K, K))
        lw2 = log_importance_weight((B_coef, Sigma), Q)
    finally:
        ident._REF_CACHE.clear()
        ident._REF_CACHE.update(saved)
    assert lw1 == pytest.approx(lw2, abs=1e-6)


def test_g_fh_is_finite_difference_continuous():
    """Our g.f map must be smooth at FD scale (bsvarSIGNs' C++ g_fh is not:
    measured jumps of order 1e6*h from Armadillo's SVD null(), which is why
    its volume element was not usable as a numerical benchmark)."""
    rng = np.random.default_rng(3)
    B_coef, Sigma, Q = _random_case(rng, m=0)
    L = np.linalg.cholesky(Sigma)
    A0 = np.linalg.solve(L.T, Q)
    x = A0.reshape(-1, order="F")
    g0 = _g_fh(x, K, 0)
    h = 1e-6
    for i in [0, 5, 20, 40, 63]:
        xp = x.copy()
        xp[i] += h
        fd = np.abs(_g_fh(xp, K, 0) - g0) / h
        assert fd.max() < 1e3, f"g.f jumps at coordinate {i}: {fd.max():.1e}"


def _bsvarsigns_available() -> bool:
    import shutil
    import subprocess
    if shutil.which("Rscript") is None:
        return False
    try:
        r = subprocess.run(
            ["Rscript", "-e",
             'cat(requireNamespace("bsvarSIGNs", quietly=TRUE))'],
            capture_output=True, text=True, timeout=60)
        return "TRUE" in r.stdout
    except Exception:
        return False


@pytest.mark.skipif(not _bsvarsigns_available(),
                    reason="Rscript with bsvarSIGNs not available")
def test_bsvarsigns_cross_check(tmp_path):
    """Compare against the bsvarSIGNs C++ reference on an identical draw.

    The analytically closed-form piece (log ve_f = -(2n+m+1) log|det A0|)
    must agree exactly. The volume-element piece CANNOT be value-matched:
    bsvarSIGNs' g_fh uses Armadillo's SVD null(), whose basis jumps O(1)
    under the finite-difference steps of its Jacobian, so its volume element
    is dominated by basis-jump noise; the test asserts that measured
    discontinuity (finite differences of order >= 1e3) as the documented
    reason for the deviation. Our smooth-basis volume element is instead
    verified against the independent explicit parameterization above.
    """
    import json
    import subprocess

    from boe_var.identification import _shock_order

    rng = np.random.default_rng(21)
    B_coef, Sigma, Q = _random_case(rng, m=2)
    L = np.linalg.cholesky(Sigma)
    A0 = np.linalg.solve(L.T, Q)
    _, ldA0 = np.linalg.slogdet(A0)
    log_ve_f_py = -(2 * K + B_coef.shape[0] + 1) * ldA0

    order = _shock_order(ZERO_RESTRICTIONS)
    payload = {
        "Sigma": Sigma.tolist(), "B": B_coef.tolist(), "Q": Q.tolist(),
        "order": order,
        "zeros": [ZERO_RESTRICTIONS[SHOCKS[j]] for j in order],
    }
    jf = tmp_path / "case.json"
    jf.write_text(json.dumps(payload))
    rscript = tmp_path / "check.R"
    rscript.write_text(r'''
args <- commandArgs(trailingOnly = TRUE)
library(jsonlite)
d <- fromJSON(args[1], simplifyMatrix = TRUE, simplifyDataFrame = FALSE)
ns <- getNamespace("bsvarSIGNs")
gfh <- get("_bsvarSIGNs_g_fh_vec", envir = ns)
N <- nrow(d$Sigma)
order <- unlist(d$order) + 1
Qp <- d$Q[, order]
U <- chol(d$Sigma)
A0 <- solve(U) %*% Qp
Aplus <- d$B %*% A0
Z <- list()
for (i in seq_along(order)) {
  vars <- d$zeros[[i]]
  if (length(vars) == 0) Z[[i]] <- matrix(0, 0, N)
  else { M <- matrix(0, length(vars), N)
         for (r in seq_along(vars)) M[r, vars[[r]] + 1] <- 1
         Z[[i]] <- M }
}
lvf <- -(2 * N + nrow(d$B) + 1) * determinant(A0)$modulus[1]
x <- c(as.vector(A0), as.vector(Aplus))
g0 <- .Call(gfh, Z, x)
h <- 1e-6
jump <- 0
for (i in c(1, 5, 20, 40)) {
  xp <- x; xp[i] <- xp[i] + h
  jump <- max(jump, max(abs(.Call(gfh, Z, xp) - g0)) / h)
}
cat(sprintf("%.12f %.6e\n", lvf, jump))
''')
    r = subprocess.run(["Rscript", str(rscript), str(jf)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    lvf_r, jump_r = map(float, r.stdout.split()[-2:])
    # Closed-form piece agrees exactly (sign convention: |det A0|).
    assert log_ve_f_py == pytest.approx(lvf_r, abs=1e-8)
    # Documented discontinuity of the C++ g_fh under FD perturbation.
    assert jump_r > 1e3

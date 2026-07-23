"""Zero + sign restriction identification (Arias, Rubio-Ramirez & Waggoner, 2018).

Encodes Table 2 of Brignone & Piffer (2025), impact restrictions only, and
implements the ARRW recursive construction of orthogonal Q such that
B = chol(Sigma) @ Q satisfies the zero restrictions exactly; sign
restrictions are imposed by accept/reject.
"""

from __future__ import annotations

import numpy as np

# Variable order (rows of B)
VARIABLES = [
    "world_gdp",
    "world_cpi",
    "oil_price",
    "bank_rate",
    "eri",
    "cpisa",
    "cpi_energy",
    "uk_gdp",
]

# Shock order (columns of B)
SHOCKS = [
    "world_demand",
    "world_energy",
    "world_supply",
    "unident_global",
    "uk_demand",
    "uk_supply",
    "uk_monpol",
    "unident_uk",
]

K = len(VARIABLES)

# Zero restrictions on impact: shock -> variable indices with zero response.
# UK shocks (incl. unident_uk) have zero impact on the world block
# (world_gdp, world_cpi, oil_price); world_supply has zero impact on
# cpi_energy; unident_global carries no restrictions at all.
ZERO_RESTRICTIONS: dict[str, list[int]] = {
    "world_demand": [],
    "world_energy": [],
    "world_supply": [6],          # cpi_energy
    "unident_global": [],
    "uk_demand": [0, 1, 2],       # world_gdp, world_cpi, oil_price
    "uk_supply": [0, 1, 2],
    "uk_monpol": [0, 1, 2],
    "unident_uk": [0, 1, 2],
}

# Sign restrictions on impact: shock -> list of (variable_index, sign).
SIGN_RESTRICTIONS: dict[str, list[tuple[int, int]]] = {
    "world_demand": [(0, +1), (1, +1), (5, +1), (6, +1), (7, +1)],
    "world_energy": [(0, +1), (1, -1), (2, -1), (5, -1), (6, -1), (7, +1)],
    "world_supply": [(0, +1), (1, -1), (5, -1), (7, +1)],
    "unident_global": [],
    "uk_demand": [(3, +1), (5, +1), (7, +1)],
    "uk_supply": [(5, -1), (7, +1)],
    "uk_monpol": [(3, +1), (5, -1), (7, -1)],
    "unident_uk": [],
}


def _shock_order(zero_restrictions: dict[str, list[int]]) -> list[int]:
    """Shock (column) indices ordered by number of zero restrictions, most first.

    Ties broken by original position for determinism. The ARRW recursive
    algorithm requires processing columns with more zero restrictions first.
    """
    return sorted(range(K), key=lambda j: (-len(zero_restrictions[SHOCKS[j]]), j))


def draw_Q(
    Sigma: np.ndarray,
    rng: np.random.Generator | None = None,
    zero_restrictions: dict[str, list[int]] | None = None,
) -> np.ndarray:
    """Draw orthogonal Q such that B = chol(Sigma) @ Q satisfies the zero restrictions.

    For column q_j of Q, the zero restriction B[v, j] = 0 reads
    L[v, :] @ q_j = 0 where L = chol(Sigma). Following Arias et al. (2018),
    columns are drawn recursively (most-restricted first): draw z ~ N(0, I),
    project onto the null space of the matrix stacking the relevant rows of L
    and the previously drawn columns, then normalize.
    """
    if rng is None:
        rng = np.random.default_rng()
    if zero_restrictions is None:
        zero_restrictions = ZERO_RESTRICTIONS

    L = np.linalg.cholesky(np.asarray(Sigma, dtype=float))
    order = _shock_order(zero_restrictions)

    Q = np.zeros((K, K))
    drawn: list[np.ndarray] = []
    for j in order:
        rows = zero_restrictions[SHOCKS[j]]
        constraints = [L[v, :] for v in rows] + drawn
        z = rng.standard_normal(K)
        if constraints:
            M = np.vstack(constraints)
            # Project z onto the null space of M.
            # Use SVD-based null-space basis for numerical robustness.
            _, s, Vt = np.linalg.svd(M)
            rank = int(np.sum(s > 1e-12 * s[0])) if s.size else 0
            N = Vt[rank:].T  # K x (K - rank), orthonormal null-space basis
            if N.shape[1] == 0:
                raise ValueError(
                    f"No feasible direction for shock '{SHOCKS[j]}': "
                    "too many zero restrictions."
                )
            z = N @ (N.T @ z)
        nrm = np.linalg.norm(z)
        if nrm < 1e-12:  # pragma: no cover - probability zero
            raise RuntimeError("Degenerate draw; retry.")
        q = z / nrm
        Q[:, j] = q
        drawn.append(q)
    return Q


def check_signs(
    B: np.ndarray,
    sign_restrictions: dict[str, list[tuple[int, int]]] | None = None,
) -> np.ndarray | None:
    """Check sign restrictions on impact matrix B, allowing column sign flips.

    For each shock column with sign restrictions, accept if all restrictions
    hold either for the column as drawn or for its negation (standard
    normalization); the flipped column is used in the returned matrix.
    Returns the (possibly flipped) B if all columns pass, else None.
    """
    if sign_restrictions is None:
        sign_restrictions = SIGN_RESTRICTIONS
    B = np.asarray(B, dtype=float).copy()
    for j, shock in enumerate(SHOCKS):
        restr = sign_restrictions.get(shock, [])
        if not restr:
            continue
        col = B[:, j]
        if all(s * col[v] > 0 for v, s in restr):
            continue
        if all(s * (-col[v]) > 0 for v, s in restr):
            B[:, j] = -col
            continue
        return None
    return B


def draw_B(
    Sigma: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """One candidate B = chol(Sigma) @ draw_Q(...); None if signs rejected."""
    Q = draw_Q(Sigma, rng)
    B = np.linalg.cholesky(np.asarray(Sigma, dtype=float)) @ Q
    return check_signs(B)


# ---------------------------------------------------------------------------
# Chan-Matthes-Yu (2025) blockwise permutation search (arXiv 2503.20668,
# Algorithm 1 + Proposition 1, adapted to zero restrictions).
#
# The Haar measure is invariant under column permutations and sign flips, but
# with zero restrictions a column may only be relabelled to a shock position
# with the IDENTICAL zero-restriction set. Blocks of exchangeable shocks:
#   - {world_demand, world_energy, unident_global}: no zeros
#   - {world_supply}: singleton (zero on cpi_energy)
#   - {uk_demand, uk_supply, uk_monpol, unident_uk}: zeros on vars 0, 1, 2
# ---------------------------------------------------------------------------

BLOCKS: list[list[int]] = [[0, 1, 3], [2], [4, 5, 6, 7]]

_WILDCARD = 2  # T-matrix marker: unrestricted shock accepts any column, any sign


def _block_T(B: np.ndarray, block: list[int], sign_restrictions) -> np.ndarray:
    """T[j, i] over a block: +1 if column block[i] satisfies shock block[j]'s
    sign restrictions as-is, -1 if the negated column does, _WILDCARD if the
    shock is unrestricted (accepts anything), 0 otherwise."""
    nb = len(block)
    T = np.zeros((nb, nb), dtype=int)
    for a, j in enumerate(block):
        restr = sign_restrictions.get(SHOCKS[j], [])
        for b, i in enumerate(block):
            col = B[:, i]
            if not restr:
                T[a, b] = _WILDCARD
            elif all(s * col[v] > 0 for v, s in restr):
                T[a, b] = +1
            elif all(s * col[v] < 0 for v, s in restr):
                T[a, b] = -1
    return T


def permutation_search(
    B: np.ndarray,
    rng: np.random.Generator | None = None,
    sign_restrictions: dict[str, list[tuple[int, int]]] | None = None,
) -> np.ndarray | None:
    """Search within-block column permutations / sign flips for a valid B.

    Within each block (shocks sharing identical zero-restriction sets, so
    relabelling preserves the zeros exactly), all permutations are enumerated
    (blocks have size <= 4); a perfect matching assigns each shock a distinct
    column whose T-entry is nonzero. If several matchings exist one is chosen
    uniformly at random; unrestricted shocks take leftover columns with a
    random +-1 flip. Returns the reassembled B, or None if some block admits
    no valid matching (joint rejection).
    """
    from itertools import permutations

    if rng is None:
        rng = np.random.default_rng()
    if sign_restrictions is None:
        sign_restrictions = SIGN_RESTRICTIONS
    B = np.asarray(B, dtype=float)
    out = np.empty_like(B)

    for block in BLOCKS:
        nb = len(block)
        T = _block_T(B, block, sign_restrictions)
        valid = [
            p for p in permutations(range(nb))
            if all(T[a, p[a]] != 0 for a in range(nb))
        ]
        if not valid:
            return None
        perm = valid[int(rng.integers(len(valid)))]
        for a, j in enumerate(block):
            i = block[perm[a]]
            t = T[a, perm[a]]
            sign = float(t) if t in (1, -1) else float(rng.choice([-1.0, 1.0]))
            out[:, j] = sign * B[:, i]

    # Zeros must survive the relabelling (blocks share zero sets).
    for j, shock in enumerate(SHOCKS):
        for v in ZERO_RESTRICTIONS[shock]:
            assert abs(out[v, j]) < 1e-8, "zero restriction violated after permutation"
    return out


# ---------------------------------------------------------------------------
# Arias, Rubio-Ramirez & Waggoner (2018) importance weights for zero
# restrictions, ported from bsvarSIGNs src/restrictions_zero.cpp
# (zero_restrictions / g_fh / log_volume_element / weight_zero).
#
# Deviations from the C++ reference (documented):
#   * Sigma enters the g.f map through vech(Sigma) (its n(n+1)/2 free
#     entries) rather than the full vec(Sigma) used by bsvarSIGNs. vech is
#     the theoretically correct parameterization of the symmetric Sigma
#     manifold (Arias et al. 2018) and makes D_N exactly square:
#     dim(x) - #zeros rows and columns.
#   * Null-space bases are built from a complete Householder QR of M' (not
#     SVD): within the null space the SVD basis is attached to (numerically)
#     zero singular values and can rotate arbitrarily under the tiny
#     perturbations of finite differencing, while the pivot-free QR basis is
#     deterministic and locally smooth. The volume element is invariant to
#     the (smooth, deterministic) orthonormal basis choice, and draw_Q only
#     uses the basis through the invariant projector N N', so both remain
#     consistent with draw_Q's construction.
# ---------------------------------------------------------------------------


def _null_qr(M: np.ndarray, n: int) -> np.ndarray:
    """Orthonormal null-space basis of M (r x n, full row rank) via complete
    QR of M'. Used only at base points (never differentiated through), so
    LAPACK's discrete basis choices are harmless here."""
    if M.size == 0:
        return np.eye(n)
    Qf, _ = np.linalg.qr(M.T, mode="complete")
    return Qf[:, M.shape[0]:]


def _null_smooth(M: np.ndarray, n: int, ref: np.ndarray) -> np.ndarray:
    """Orthonormal null-space basis of M that is a SMOOTH function of M.

    A complete-QR (or SVD) null basis can jump discretely under tiny
    perturbations of M (LAPACK branch/sign choices), which wrecks the
    finite-difference Jacobian of the g.f map. Instead, project the fixed
    reference vectors `ref` onto null(M) with the smooth projector
    P = I - M'(MM')^{-1}M and Gram-Schmidt them via reduced QR, sign-fixed so
    that diag(R) > 0 -- the unique (hence locally smooth) orthonormalization.
    The volume element is invariant to the orthonormal basis choice.
    """
    dim = n - M.shape[0] if M.size else n
    if M.size == 0:
        G = ref[:, :dim]
    else:
        P = np.eye(n) - M.T @ np.linalg.solve(M @ M.T, M)
        G = P @ ref[:, :dim]
    Qr, R = np.linalg.qr(G)
    return Qr * np.sign(np.diag(R))


def _n_zero_restrictions() -> int:
    return sum(len(v) for v in ZERO_RESTRICTIONS.values())


def _unpack_x(x: np.ndarray, n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    A0 = x[: n * n].reshape(n, n, order="F")
    Aplus = x[n * n:].reshape(m, n, order="F")
    return A0, Aplus


def _zero_fn(x: np.ndarray, n: int, m: int) -> np.ndarray:
    """z(x): the impact-matrix entries restricted to zero, impact = inv(A0)'."""
    A0, _ = _unpack_x(x, n, m)
    impact = np.linalg.inv(A0).T
    order = _shock_order(ZERO_RESTRICTIONS)
    return np.array(
        [impact[v, j] for j in order for v in ZERO_RESTRICTIONS[SHOCKS[j]]]
    )


def _g_fh(x: np.ndarray, n: int, m: int) -> np.ndarray:
    """g.f: x = [vec(A0); vec(A+)] -> [vec(B_coef); vech(Sigma); w_1..w_n].

    w_j are the sphere coordinates of the recursive draw_Q construction, in
    the same processing order (most zeros first): w_j = K_j' q_j, K_j an
    orthonormal basis of the null space of [zero-constraint rows of L;
    previously placed columns of Q], reconstructed deterministically from
    (Sigma, previous columns) exactly as draw_Q's projection subspace.
    """
    A0, Aplus = _unpack_x(x, n, m)
    A0_inv = np.linalg.inv(A0)
    B_coef = Aplus @ A0_inv
    Sigma = np.linalg.inv(A0 @ A0.T)
    Sigma = 0.5 * (Sigma + Sigma.T)
    L = np.linalg.cholesky(Sigma)
    Q = L.T @ A0

    ref = _ref_matrix(n)
    parts = [B_coef.reshape(-1, order="F"), Sigma[np.tril_indices(n)]]
    order = _shock_order(ZERO_RESTRICTIONS)
    prev: list[np.ndarray] = []
    for j in order:
        rows = [L[v, :] for v in ZERO_RESTRICTIONS[SHOCKS[j]]] + prev
        M = np.vstack(rows) if rows else np.empty((0, n))
        Kj = _null_smooth(M, n, ref)
        parts.append(Kj.T @ Q[:, j])
        prev.append(Q[:, j])
    return np.concatenate(parts)


_REF_CACHE: dict[int, np.ndarray] = {}


def _ref_matrix(n: int) -> np.ndarray:
    """Fixed (deterministic) reference vectors for the smooth null bases."""
    if n not in _REF_CACHE:
        _REF_CACHE[n] = np.random.default_rng(20180101).standard_normal((n, n))
    return _REF_CACHE[n]


def _num_jacobian(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of f at x."""
    f0 = f(x)
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        step = h * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += step
        xm = x.copy()
        xm[i] -= step
        J[:, i] = (f(xp) - f(xm)) / (2.0 * step)
    return J


def log_importance_weight(draw_or_coefs, Q: np.ndarray) -> float:
    """Log Arias et al. (2018) importance weight for an accepted (draw, Q).

    Parameters
    ----------
    draw_or_coefs : PosteriorDraw-like with .Pi (k x m) and .Sigma, or a
        tuple (B_coef, Sigma) with B_coef the m x n reduced-form coefficient
        matrix (B_coef = draw.Pi.T; m may be 0).
    Q : the FINAL (relabelled) orthogonal matrix from the permutation step.

    Returns log w = log ve_f - log ve_gf with
    log ve_f = -(2n+m+1) log|det A0|, A0 = inv(chol(Sigma,lower)') Q, and
    log ve_gf = 0.5 log det(D_N' D_N), D_N = D_{g.f} @ null(D_z).
    """
    if isinstance(draw_or_coefs, tuple):
        B_coef, Sigma = draw_or_coefs
        B_coef = np.asarray(B_coef, dtype=float)
    else:
        Sigma = draw_or_coefs.Sigma
        Pi = getattr(draw_or_coefs, "Pi", None)
        B_coef = np.zeros((0, K)) if Pi is None else np.asarray(Pi, float).T
    Sigma = np.asarray(Sigma, dtype=float)
    n = Sigma.shape[0]
    m = B_coef.shape[0]

    L = np.linalg.cholesky(Sigma)
    A0 = np.linalg.solve(L.T, Q)          # inv(chol') @ Q
    Aplus = B_coef @ A0
    x = np.concatenate(
        [A0.reshape(-1, order="F"), Aplus.reshape(-1, order="F")]
    )

    n_zeros = _n_zero_restrictions()
    d = x.size                            # n(n+m)
    gf0 = _g_fh(x, n, m)
    # Dimension consistency (Arias et al. 2018): each w_j is a unit vector
    # (one redundant coordinate), so the image manifold has dimension
    # mn + n(n+1)/2 + sum_j (dim(w_j) - 1) which must equal dim(x) - #zeros.
    # D_N is therefore tall (gf0.size x (d - n_zeros)) with full column rank,
    # exactly as in bsvarSIGNs.
    assert gf0.size - n == d - n_zeros, (gf0.size, n, d, n_zeros)

    Dz = _num_jacobian(lambda v: _zero_fn(v, n, m), x)
    Dgf = _num_jacobian(lambda v: _g_fh(v, n, m), x)
    Nz = _null_qr(Dz, d)
    DN = Dgf @ Nz

    sign, logdet_A0 = np.linalg.slogdet(A0)
    log_ve_f = -(2 * n + m + 1) * logdet_A0   # log|det A0| (sign dropped)
    _, ld = np.linalg.slogdet(DN.T @ DN)
    log_ve_gf = 0.5 * ld
    return float(log_ve_f - log_ve_gf)


def importance_weight(draw_or_coefs, Q: np.ndarray) -> float:
    """exp of :func:`log_importance_weight`; normalize across draws downstream.

    Weights are only defined up to a common constant, so compute in log
    space, subtract the max across accepted draws before exponentiating if
    overflow is a concern (identify() does this).
    """
    return float(np.exp(log_importance_weight(draw_or_coefs, Q)))


def ess(weights) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2."""
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    return float(s * s / (w * w).sum()) if w.size else 0.0


# Below these thresholds, weighted quantile bands are typically noisy and
# asymmetric; results should carry an explicit warning.
MIN_RELIABLE_ACCEPTED = 100
MIN_RELIABLE_ESS = 100.0


def weak_inference_warnings(n_accepted: int, ess_value: float,
                            n_draws: int | None = None) -> list[str]:
    """Human-readable warnings when identification output is too thin.

    Returns a (possibly empty) list of warning strings. Emitted when fewer
    than ``MIN_RELIABLE_ACCEPTED`` draws survive the sign-restriction
    accept/reject or the importance-weight ESS falls below
    ``MIN_RELIABLE_ESS``. Includes a concrete draw-count recommendation
    scaled from the observed acceptance/ESS yield when ``n_draws`` is given.
    """
    warnings: list[str] = []
    rec = None
    if n_draws and n_draws > 0:
        # Draws needed so that the scarcer of (accepted, ESS) reaches its
        # threshold at the observed yield, with 50% headroom.
        yields = []
        if n_accepted > 0:
            yields.append(MIN_RELIABLE_ACCEPTED / (n_accepted / n_draws))
            if ess_value > 0:
                yields.append(MIN_RELIABLE_ESS / (ess_value / n_draws))
        if yields:
            rec = int(np.ceil(1.5 * max(yields) / 100.0) * 100)
    suffix = (f" Re-run with draws >= {max(rec, n_draws or 0)}."
              if rec is not None else " Re-run with a higher draw count.")
    if n_accepted < MIN_RELIABLE_ACCEPTED:
        warnings.append(
            f"Weak identification sample: only {n_accepted} accepted draws "
            f"(< {MIN_RELIABLE_ACCEPTED}); posterior bands may be noisy and "
            f"asymmetric." + suffix
        )
    if ess_value < MIN_RELIABLE_ESS:
        warnings.append(
            f"Low importance-weight effective sample size (ESS = "
            f"{ess_value:.1f} < {MIN_RELIABLE_ESS:.0f}); weighted quantiles "
            f"are dominated by few draws and may be unreliable." + suffix
        )
    return warnings


def identify(
    posterior_draws,
    rng: np.random.Generator | None = None,
    target_accepted: int | None = None,
    max_tries_per_draw: int | None = None,  # deprecated, ignored (one Q per draw)
    compute_weights: bool = True,
) -> list[tuple[object, np.ndarray, float]]:
    """Identify structural impact matrices with joint (Sigma, Q) accept/reject.

    Exactly ONE Q is attempted per posterior draw (Arias, Rubio-Ramirez,
    Shin & Waggoner 2024: per-draw retries target the wrong distribution).
    After the Chan-Matthes-Yu blockwise permutation search, failures discard
    the posterior draw entirely. Returns a list of (draw, B, w) triples where
    w is the Arias et al. (2018) zero-restriction importance weight
    (normalized to mean 1 across accepted draws; all 1.0 if
    compute_weights=False). The acceptance fraction is logged and stored in
    ``identify.last_acceptance_rate``.

    ``max_tries_per_draw`` is accepted for backwards compatibility but
    ignored; ``target_accepted`` optionally caps the number of accepted
    draws.
    """
    import logging

    if rng is None:
        rng = np.random.default_rng()
    accepted: list[tuple[object, np.ndarray]] = []
    log_ws: list[float] = []
    attempted = 0
    for draw in posterior_draws:
        if target_accepted is not None and len(accepted) >= target_accepted:
            break
        attempted += 1
        Q = draw_Q(draw.Sigma, rng)
        L = np.linalg.cholesky(np.asarray(draw.Sigma, dtype=float))
        B = permutation_search(L @ Q, rng)
        if B is None:
            continue  # joint rejection of (Sigma, Q)
        if compute_weights:
            Q_final = np.linalg.solve(L, B)  # relabelled Q
            log_ws.append(log_importance_weight(draw, Q_final))
        else:
            log_ws.append(0.0)
        accepted.append((draw, B))

    rate = len(accepted) / attempted if attempted else 0.0
    identify.last_acceptance_rate = rate
    logging.getLogger(__name__).info(
        "identify: accepted %d/%d posterior draws (%.1f%%)",
        len(accepted), attempted, 100 * rate,
    )
    if not accepted:
        identify.last_diagnostics = {
            "attempted": attempted, "accepted": 0, "acceptance_rate": rate,
            "ess": 0.0,
            "warnings": weak_inference_warnings(0, 0.0, attempted),
        }
        return []
    lw = np.asarray(log_ws)
    w = np.exp(lw - lw.max())
    w *= len(w) / w.sum()  # normalize to mean 1
    ess_value = ess(w)
    identify.last_diagnostics = {
        "attempted": attempted,
        "accepted": len(accepted),
        "acceptance_rate": rate,
        "ess": ess_value,
        "warnings": weak_inference_warnings(len(accepted), ess_value,
                                            attempted),
    }
    for msg in identify.last_diagnostics["warnings"]:
        logging.getLogger(__name__).warning("identify: %s", msg)
    return [(draw, B, float(wi)) for (draw, B), wi in zip(accepted, w)]


identify.last_acceptance_rate = None
identify.last_diagnostics = None


def structural_shocks(B: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Structural shocks eps_t = inv(B) @ u_t.

    `residuals` is (T, k) with rows u_t'; returns (T, k) with rows eps_t'.
    """
    return np.linalg.solve(B, np.asarray(residuals, dtype=float).T).T

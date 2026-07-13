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


def identify(
    posterior_draws,
    target_accepted: int = 1000,
    max_tries_per_draw: int = 100,
    rng: np.random.Generator | None = None,
) -> list[tuple[object, np.ndarray]]:
    """Identify structural impact matrices for a set of posterior draws.

    Cycles through `posterior_draws` (objects with a `.Sigma` attribute); for
    each draw, attempts up to `max_tries_per_draw` Q draws until one satisfies
    the sign restrictions. Returns a list of (draw, B) pairs with at most
    `target_accepted` elements (fewer if the draw budget is exhausted).
    """
    if rng is None:
        rng = np.random.default_rng()
    accepted: list[tuple[object, np.ndarray]] = []
    for draw in posterior_draws:
        if len(accepted) >= target_accepted:
            break
        for _ in range(max_tries_per_draw):
            B = draw_B(draw.Sigma, rng)
            if B is not None:
                accepted.append((draw, B))
                break
    return accepted


def structural_shocks(B: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Structural shocks eps_t = inv(B) @ u_t.

    `residuals` is (T, k) with rows u_t'; returns (T, k) with rows eps_t'.
    """
    return np.linalg.solve(B, np.asarray(residuals, dtype=float).T).T

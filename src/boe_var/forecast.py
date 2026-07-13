"""Forecasting machinery and Section 5 forecast-revision tools.

Implements the paper's (Brignone & Piffer 2025, pp. 19-24) forecasting
objects: unconditional (deterministic) and stochastic forecasts, predictive
bands across posterior draws, the posterior distribution of the period-T
structural shocks, the composite impulse response sum_j eps_{T,j} IRF_j, and
the Figure 9 forecast-revision identity

    yhat_{T+h | T} = yhat_{T+h | T-1} + Psi_h B eps_T        (exact per draw)

which holds because the forecast is linear in the state and
u_T = y_T - Pi x_T = B eps_T.

All functions accept a ``PosteriorDraw``-like object with attributes ``.Pi``
(k x m, regressor layout [lags | constant | dummies]), ``.Sigma``, ``.lags``,
``.k`` and method ``.companion()`` (see boe_var.bvar).
"""

from __future__ import annotations

import numpy as np

from .analysis import _ma_coeffs, aggregate

__all__ = [
    "unconditional_forecast",
    "sample_forecast",
    "forecast_bands",
    "yoy",
    "reduced_form_residuals",
    "shock_distribution_T",
    "composite_irf",
    "yoy_transform_irf",
    "forecast_revision",
]


# ---------------------------------------------------------------------------
# Point / stochastic forecasts
# ---------------------------------------------------------------------------

def _dims(draw) -> tuple[int, int, int]:
    """(k, p, n_dummies) inferred from the draw."""
    Pi = np.asarray(draw.Pi, dtype=float)
    k, p = draw.k, draw.lags
    n_d = Pi.shape[1] - k * p - 1
    if n_d < 0:
        raise ValueError("Pi has fewer columns than k*lags + 1")
    return k, p, n_d


def _iterate(draw, y_hist, horizons, shocks=None, future_dummies=None):
    """Iterate y_{t+h} = Pi x + (shock) from the last p rows of ``y_hist``.

    Future dummies are zero unless ``future_dummies`` (H x n_d) is given.
    ``shocks``: optional (H, k) innovations added each step.
    Returns (H, k).
    """
    Pi = np.asarray(draw.Pi, dtype=float)
    k, p, n_d = _dims(draw)
    y_hist = np.asarray(y_hist, dtype=float)
    if y_hist.ndim != 2 or y_hist.shape[1] != k:
        raise ValueError("y_hist must be (T, k)")
    if y_hist.shape[0] < p:
        raise ValueError("y_hist must contain at least `lags` rows")

    lags = [y_hist[-l] for l in range(1, p + 1)]  # most recent first
    out = np.empty((horizons, k))
    for h in range(horizons):
        d = np.zeros(n_d) if future_dummies is None else \
            np.asarray(future_dummies, dtype=float)[h]
        x = np.concatenate(lags + [[1.0], d])
        y_next = Pi @ x
        if shocks is not None:
            y_next = y_next + shocks[h]
        out[h] = y_next
        lags = [y_next] + lags[:-1] if p > 1 else [y_next]
    return out


def unconditional_forecast(draw, y_hist, dummies_hist=None, horizons: int = 13):
    """Deterministic forecast path (zero future shocks, future dummies = 0).

    ``y_hist``: (T, k) data through the forecast origin; only the last p
    rows matter. ``dummies_hist`` is accepted for interface symmetry (the
    historical dummies do not enter the forecast; future dummies are zero).
    Returns (horizons, k): rows are yhat_{T+1}, ..., yhat_{T+H}.
    """
    del dummies_hist  # historical dummies do not affect the projection
    return _iterate(draw, y_hist, horizons)


def sample_forecast(draw, y_hist, dummies_hist=None, horizons: int = 13,
                    rng: np.random.Generator | None = None):
    """One stochastic forecast path: adds u_{T+h} ~ N(0, Sigma) each step.

    Combines with sampling across posterior draws to give the full
    predictive distribution (parameter + shock uncertainty).
    """
    del dummies_hist
    if rng is None:
        rng = np.random.default_rng()
    Sigma = np.asarray(draw.Sigma, dtype=float)
    k = draw.k
    # Robust to (numerically) singular / zero Sigma: use eigen square root.
    w, V = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    Ls = V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    shocks = rng.standard_normal((horizons, k)) @ Ls.T
    return _iterate(draw, y_hist, horizons, shocks=shocks)


def _unpack_draw(item):
    """Accept a draw, (draw, B) pair, or (draw, B, w) triple."""
    return item[0] if isinstance(item, (tuple, list)) else item


def forecast_bands(pairs_or_draws, y_hist, dummies_hist=None,
                   horizons: int = 13, weights=None, n_paths: int = 5,
                   rng: np.random.Generator | None = None) -> dict:
    """Pointwise median + 68/90 bands of the predictive distribution.

    Samples ``n_paths`` stochastic paths per posterior draw (parameter +
    shock uncertainty). ``weights``: one importance weight per draw,
    replicated across that draw's paths. Returns the ``analysis.aggregate``
    dict of (horizons, k) arrays.
    """
    if rng is None:
        rng = np.random.default_rng()
    paths, w = [], []
    weights = None if weights is None else np.asarray(weights, dtype=float)
    for i, item in enumerate(pairs_or_draws):
        draw = _unpack_draw(item)
        for _ in range(n_paths):
            paths.append(sample_forecast(draw, y_hist, dummies_hist,
                                         horizons, rng))
            w.append(1.0 if weights is None else weights[i])
    return aggregate(paths, weights=None if weights is None else np.asarray(w))


def yoy(series_levels: np.ndarray) -> np.ndarray:
    """YoY growth from 100*log levels: x_t - x_{t-4} along axis 0.

    Since the data are 100*log levels, the 4-quarter log difference is
    directly YoY growth in percent. Output loses the first 4 rows.
    """
    x = np.asarray(series_levels, dtype=float)
    if x.shape[0] <= 4:
        raise ValueError("need more than 4 observations for YoY")
    return x[4:] - x[:-4]


# ---------------------------------------------------------------------------
# Residuals / shocks beyond the estimation sample
# ---------------------------------------------------------------------------

def reduced_form_residuals(draw, y, dummies=None) -> np.ndarray:
    """u_t = y_t - Pi x_t over rows lags..T-1 of ``y`` (any sample).

    ``dummies``: (T, n_d) aligned with ``y`` (zeros where no dummy is on,
    e.g. after 2021Q2); if None, zeros are used.
    """
    y = np.asarray(y, dtype=float)
    Pi = np.asarray(draw.Pi, dtype=float)
    k, p, n_d = _dims(draw)
    T = y.shape[0]
    if dummies is None:
        dummies = np.zeros((T, n_d))
    dummies = np.asarray(dummies, dtype=float)
    rows = []
    for t in range(p, T):
        xlags = np.concatenate([y[t - l] for l in range(1, p + 1)])
        rows.append(np.concatenate([xlags, [1.0], dummies[t]]))
    X = np.asarray(rows)
    return y[p:] - X @ Pi.T


def shock_distribution_T(pairs, resid_fn_or_shocks, weights=None,
                         t: int = -1) -> dict:
    """Posterior distribution of the structural shocks at one period (paper Fig 7).

    ``pairs``: iterable of (draw, B) or (draw, B, w).
    ``resid_fn_or_shocks``: either a callable ``draw -> (T, k)`` reduced-form
    residuals (row ``t`` is used, default the last in-sample period T), or a
    precomputed (n_pairs, k) array of structural shocks eps_T.
    Returns dict with ``samples`` (n, k) eps_T per accepted draw, ``weights``
    (n,), and ``p_pos`` (k,) weighted P(shock > 0).
    """
    pairs = list(pairs)
    n = len(pairs)
    if callable(resid_fn_or_shocks):
        eps = np.empty((n, pairs[0][0].k))
        for i, item in enumerate(pairs):
            draw, B = item[0], item[1]
            u_t = np.asarray(resid_fn_or_shocks(draw), dtype=float)[t]
            eps[i] = np.linalg.solve(np.asarray(B, dtype=float), u_t)
    else:
        eps = np.asarray(resid_fn_or_shocks, dtype=float)
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    p_pos = (w[:, None] * (eps > 0)).sum(axis=0)
    return {"samples": eps, "weights": w, "p_pos": p_pos}


# ---------------------------------------------------------------------------
# Composite impulse response (paper Fig 8)
# ---------------------------------------------------------------------------

def composite_contributions(draw, B, eps_T, horizons: int):
    """(H, k_var, k_shock): contribution of shock j to variable i at horizon h,
    i.e. Psi_h[i, j] * eps_{T, j} (Psi_h = J F^h J' B, unit-sd shocks)."""
    Psi = _ma_coeffs(draw, B, horizons)          # (H, k, k)
    return Psi * np.asarray(eps_T, dtype=float)[np.newaxis, np.newaxis, :]


def composite_irf(pairs, weights=None, eps_T=None, horizons: int = 13,
                  resid_fn=None) -> dict:
    """Composite impulse response sum_j eps_{T,j} * IRF_j per draw, aggregated.

    Each draw uses its OWN eps_T and IRFs. ``eps_T``: (n_pairs, k) array of
    per-draw structural shocks at T; alternatively pass ``resid_fn`` (callable
    draw -> residuals incl. period T) and eps_T is computed per draw.

    Returns dict:
      - ``joint``: aggregate dict (median/68/90 bands) of the (H, k) joint
        effect across draws;
      - ``decomp_median``: (H, k_var, k_shock) weighted median of the
        per-shock contributions (stacked-bar decomposition of the median);
      - ``contributions``: list is NOT returned (memory); recompute if needed.
    """
    pairs = list(pairs)
    if eps_T is None:
        if resid_fn is None:
            raise ValueError("provide eps_T or resid_fn")
        eps_T = shock_distribution_T(pairs, resid_fn, weights=None)["samples"]
    eps_T = np.asarray(eps_T, dtype=float)

    contribs = []
    for i, item in enumerate(pairs):
        draw, B = item[0], item[1]
        contribs.append(composite_contributions(draw, B, eps_T[i], horizons))
    stack = np.stack(contribs, axis=0)                # (n, H, k, k)
    joint = aggregate(list(stack.sum(axis=3)), weights=weights)
    from .analysis import _quantile
    decomp_median = _quantile(stack, 0.50, weights)
    return {"joint": joint, "decomp_median": decomp_median}


def yoy_transform_irf(level_effect: np.ndarray) -> np.ndarray:
    """YoY effect of a shock hitting at horizon 0 on a 100*log-level variable.

    ``level_effect``: (H, ...) level responses (0 before impact). The YoY
    (x_t - x_{t-4}) effect at horizon h is level[h] - level[h-4], with
    level = 0 for h < 0. Returns the same shape as the input.
    """
    x = np.asarray(level_effect, dtype=float)
    out = x.copy()
    if x.shape[0] > 4:
        out[4:] = x[4:] - x[:-4]
    return out


# ---------------------------------------------------------------------------
# Forecast revision (paper Fig 9)
# ---------------------------------------------------------------------------

def forecast_revision(draw, B, y_full, dummies=None, horizons: int = 13,
                      atol: float | None = None) -> dict:
    """Figure 9 logic for ONE draw, treating the last row of ``y_full`` as T.

    (i)  fc_T: forecast from data through T -> dates T+1..T+H, (H, k).
    (ii) fc_Tm1: 'pseudo' forecast from data through T-1 (same parameters)
         -> dates T, T+1, ..., T+H, (H+1, k).
    (iii) composite: Psi_h B eps_T for h = 0..H, (H+1, k), with
         eps_T = B^{-1} (y_T - Pi x_T) computed from actual data through T
         (dummies from ``dummies``, zero after the dummy window).

    Adding-up identity (exact, absent data revisions):
        y_T      = fc_Tm1[0] + composite[0]
        fc_T[h-1] = fc_Tm1[h] + composite[h],  h = 1..H.

    Returns dict with fc_T, fc_Tm1, eps_T, composite, max_err. If ``atol``
    is given, asserts max_err <= atol.
    """
    y_full = np.asarray(y_full, dtype=float)
    k, p, n_d = _dims(draw)
    B = np.asarray(B, dtype=float)

    fc_T = unconditional_forecast(draw, y_full, horizons=horizons)
    fc_Tm1 = unconditional_forecast(draw, y_full[:-1], horizons=horizons + 1)

    # eps_T from the actual regressor row at T (dummies as supplied; the
    # exercise dates are beyond the dummy window so these are zeros).
    d_T = np.zeros(n_d) if dummies is None else \
        np.asarray(dummies, dtype=float)[-1]
    x_T = np.concatenate(
        [y_full[-1 - l] for l in range(1, p + 1)] + [[1.0], d_T])
    u_T = y_full[-1] - np.asarray(draw.Pi, dtype=float) @ x_T
    eps_T = np.linalg.solve(B, u_T)

    Psi = _ma_coeffs(draw, B, horizons + 1)      # (H+1, k, k)
    composite = Psi @ eps_T                      # (H+1, k)

    recon = fc_Tm1 + composite                   # dates T..T+H
    target = np.vstack([y_full[-1], fc_T])       # y_T, then fc_T
    max_err = float(np.max(np.abs(recon - target)))
    if atol is not None:
        assert max_err <= atol, (
            f"forecast-revision adding-up identity violated: {max_err}")
    return {"fc_T": fc_T, "fc_Tm1": fc_Tm1, "eps_T": eps_T,
            "composite": composite, "max_err": max_err}

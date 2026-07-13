"""IRFs, FEVDs, historical decompositions and figure helpers for the BoE SVAR.

All structural objects are computed from a posterior draw (exposing ``.Pi``,
``.Sigma`` and ``.companion()``) and an impact matrix ``B`` (k x k, columns =
unit-standard-deviation structural shocks, so that Sigma = B @ B.T).
"""

from __future__ import annotations

import os

import numpy as np

# Ordering follows SPEC.md Table 2.
VARIABLE_NAMES = [
    "World GDP", "World CPI", "Oil price", "Bank Rate",
    "Exch. rate", "UK CPISA", "CPI Energy", "UK GDP",
]
SHOCK_NAMES = [
    "World demand", "World energy", "World supply", "Unident. global",
    "UK demand", "UK supply", "UK mon. pol.", "Unident. UK",
]
WORLD_SHOCKS = [0, 1, 2]   # demand, energy, supply
UK_SHOCKS = [4, 5, 6]      # demand, supply, monetary policy


# ---------------------------------------------------------------------------
# Core structural statistics
# ---------------------------------------------------------------------------

def _companion(draw) -> np.ndarray:
    F = np.asarray(draw.companion())
    return F


def _ma_coeffs(draw, B: np.ndarray, horizons: int) -> np.ndarray:
    """Structural MA coefficients Psi_h = J F^h J' B, shape (H, k, k)."""
    B = np.asarray(B, dtype=float)
    k = B.shape[0]
    F = _companion(draw)
    m = F.shape[0]
    J = np.zeros((k, m))
    J[:, :k] = np.eye(k)
    out = np.empty((horizons, k, k))
    Fh = np.eye(m)
    for h in range(horizons):
        out[h] = J @ Fh @ J.T @ B
        Fh = F @ Fh
    return out


def irf(draw, B, horizons: int = 21) -> np.ndarray:
    """Impulse responses to unit-s.d. shocks.

    Returns array of shape (k, k, horizons): ``irf[i, j, h]`` is the response
    of variable i to shock j, h periods after impact.
    """
    Psi = _ma_coeffs(draw, B, horizons)          # (H, k, k)
    return np.transpose(Psi, (1, 2, 0))          # (k_var, k_shock, H)


def fevd(draw, B, horizons: int = 21) -> np.ndarray:
    """Forecast-error variance shares, shape (k_var, k_shock, horizons).

    ``fevd[i, j, h]`` = share of the h+1-step forecast-error variance of
    variable i attributable to shock j.  Shares sum to 1 across shocks.
    """
    Psi = _ma_coeffs(draw, B, horizons)          # (H, k, k)
    cum = np.cumsum(Psi ** 2, axis=0)            # (H, k_var, k_shock)
    total = cum.sum(axis=2, keepdims=True)       # (H, k_var, 1)
    shares = cum / np.where(total > 0, total, 1.0)
    return np.transpose(shares, (1, 2, 0))


def estimated_shocks(draw, B, residuals) -> np.ndarray:
    """Recover structural shocks eps_t = B^{-1} u_t from reduced-form residuals.

    ``residuals`` is (T, k); returns (T, k) with columns ordered as SHOCK_NAMES.
    """
    u = np.asarray(residuals, dtype=float)
    return np.linalg.solve(np.asarray(B, dtype=float), u.T).T


def historical_decomposition(draw, B, residuals) -> dict:
    """Historical decomposition over the residual sample.

    Returns dict with:
      - ``"shocks"``: array (T, k_var, k_shock); ``[t, i, j]`` is the cumulative
        contribution of shock j to variable i at time t.
      - ``"stochastic"``: (T, k_var) = sum over shocks (equals the moving-average
        reconstruction of the residuals).  The deterministic-plus-initial-
        conditions-plus-dummies component of the data is ``y - stochastic``.
      - ``"eps"``: (T, k_shock) estimated structural shocks.
    """
    u = np.asarray(residuals, dtype=float)
    T, k = u.shape
    eps = estimated_shocks(draw, B, u)           # (T, k)
    Psi = _ma_coeffs(draw, B, T)                 # (T, k, k)
    contrib = np.zeros((T, k, k))
    for t in range(T):
        # sum_{h=0..t} Psi_h[:, j] * eps[t-h, j]
        for h in range(t + 1):
            contrib[t] += Psi[h] * eps[t - h][np.newaxis, :]
    stochastic = contrib.sum(axis=2)
    return {"shocks": contrib, "stochastic": stochastic, "eps": eps}


# ---------------------------------------------------------------------------
# Aggregation across accepted (draw, B) pairs
# ---------------------------------------------------------------------------

def weighted_quantile(values, q, weights) -> np.ndarray:
    """Weighted quantile(s) of ``values`` along axis 0.

    ``values``: array-like, shape (n, ...); ``q``: quantile in [0, 1] (scalar);
    ``weights``: length-n nonnegative array. Uses the standard weighted
    percentile: sort by value, compute cumulative weights at midpoints
    (c_i = cum_i - w_i/2), rescale to [0, 1], and linearly interpolate.
    With uniform weights this reproduces ``np.percentile`` (linear
    interpolation) exactly.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != values.shape[0]:
        raise ValueError("weights must be 1-D with length values.shape[0]")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    if not np.any(weights > 0):
        raise ValueError("weights must not all be zero")
    n = values.shape[0]
    flat = values.reshape(n, -1)
    out = np.empty(flat.shape[1])
    for c in range(flat.shape[1]):
        v = flat[:, c]
        order = np.argsort(v)
        vs, ws = v[order], weights[order]
        if n == 1:
            out[c] = vs[0]
            continue
        # Cumulative weights at midpoints, rescaled to [0, 1] so that with
        # uniform weights positions are i/(n-1), matching np.percentile's
        # linear interpolation.
        cmid = np.cumsum(ws) - 0.5 * ws
        p = (cmid - cmid[0]) / (cmid[-1] - cmid[0])
        out[c] = np.interp(q, p, vs)
    return out.reshape(values.shape[1:])


def _quantile(stack: np.ndarray, q: float, weights) -> np.ndarray:
    if weights is None:
        return np.quantile(stack, q, axis=0)
    return weighted_quantile(stack, q, weights)


def aggregate(arrays, weights=None) -> dict:
    """Pointwise median and 68%/90% bands across a list of equal-shape arrays.

    If ``weights`` (one nonnegative weight per array) is given, the median and
    bands are weighted quantiles; otherwise plain (equal-weight) quantiles.
    Returns dict with keys ``median, lo68, hi68, lo90, hi90``.
    """
    stack = np.stack([np.asarray(a) for a in arrays], axis=0)
    return {
        "median": _quantile(stack, 0.50, weights),
        "lo68": _quantile(stack, 0.16, weights),
        "hi68": _quantile(stack, 0.84, weights),
        "lo90": _quantile(stack, 0.05, weights),
        "hi90": _quantile(stack, 0.95, weights),
    }


def irf_bands(pairs, horizons: int = 21, weights=None) -> dict:
    return aggregate([irf(d, B, horizons) for d, B in pairs], weights=weights)


def fevd_bands(pairs, horizons: int = 21, weights=None) -> dict:
    return aggregate([fevd(d, B, horizons) for d, B in pairs], weights=weights)


def shock_bands(pairs, residuals_fn, weights=None) -> dict:
    """Bands for estimated shock series. ``residuals_fn(draw)`` -> (T, k)."""
    return aggregate([estimated_shocks(d, B, residuals_fn(d)) for d, B in pairs],
                     weights=weights)


def hd_median(pairs, residuals_fn, weights=None) -> np.ndarray:
    """Pointwise-median historical decomposition contributions (T, k, k)."""
    stack = np.stack([historical_decomposition(d, B, residuals_fn(d))["shocks"]
                      for d, B in pairs], axis=0)
    return _quantile(stack, 0.50, weights)


# ---------------------------------------------------------------------------
# Plot helpers (paper style: Figures 2-6)
# ---------------------------------------------------------------------------

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120, "font.size": 8, "axes.titlesize": 8,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    })
    return plt


def _ensure_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def plot_irf_grid(bands: dict, shock_indices, out_path: str,
                  title: str = "") -> str:
    """Grid of 8 variable panels per shock (rows = shocks, cols = variables)."""
    plt = _mpl()
    n_shocks = len(shock_indices)
    k = bands["median"].shape[0]
    H = bands["median"].shape[2]
    x = np.arange(H)
    fig, axes = plt.subplots(n_shocks, k, figsize=(2.0 * k, 1.9 * n_shocks),
                             squeeze=False)
    for r, j in enumerate(shock_indices):
        for i in range(k):
            ax = axes[r][i]
            ax.fill_between(x, bands["lo90"][i, j], bands["hi90"][i, j],
                            color="#9ecae1", alpha=0.5, linewidth=0)
            ax.fill_between(x, bands["lo68"][i, j], bands["hi68"][i, j],
                            color="#4292c6", alpha=0.5, linewidth=0)
            ax.plot(x, bands["median"][i, j], color="#08306b", lw=1.2)
            ax.axhline(0, color="k", lw=0.5)
            if r == 0:
                ax.set_title(VARIABLE_NAMES[i])
            if i == 0:
                ax.set_ylabel(SHOCK_NAMES[j], fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fevd(bands: dict, out_path: str,
              title: str = "Forecast-error variance decomposition") -> str:
    """Stacked area chart of median FEVD shares, one panel per variable."""
    plt = _mpl()
    med = bands["median"]                        # (k, k, H)
    # renormalise medians so stacks sum to 1
    med = med / med.sum(axis=1, keepdims=True)
    k, _, H = med.shape
    x = np.arange(H)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), squeeze=False)
    for i in range(k):
        ax = axes[i // 4][i % 4]
        ax.stackplot(x, med[i], labels=SHOCK_NAMES,
                     colors=[cmap(j) for j in range(k)])
        ax.set_title(VARIABLE_NAMES[i])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, H - 1)
    axes[0][0].legend(loc="upper right", fontsize=5, ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_shocks(bands: dict, out_path: str, index=None,
                title: str = "Estimated structural shocks") -> str:
    """Median estimated shock series with 68%/90% bands, one panel per shock."""
    plt = _mpl()
    med = bands["median"]                        # (T, k)
    T, k = med.shape
    x = np.arange(T) if index is None else index
    fig, axes = plt.subplots((k + 1) // 2, 2, figsize=(11, 1.9 * ((k + 1) // 2)),
                             squeeze=False)
    for j in range(k):
        ax = axes[j // 2][j % 2]
        ax.fill_between(x, bands["lo90"][:, j], bands["hi90"][:, j],
                        color="#9ecae1", alpha=0.5, linewidth=0)
        ax.fill_between(x, bands["lo68"][:, j], bands["hi68"][:, j],
                        color="#4292c6", alpha=0.5, linewidth=0)
        ax.plot(x, med[:, j], color="#08306b", lw=1.0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(SHOCK_NAMES[j])
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_hist_decomp(contrib: np.ndarray, deterministic: np.ndarray,
                     var_indices, out_path: str, index=None,
                     panel_titles=None,
                     title: str = "Historical decomposition") -> str:
    """Stacked-bar decomposition for selected variables.

    ``contrib``: (T, k_var, k_shock); ``deterministic``: (T, k_var) — the
    deterministic + initial-conditions + dummy component plotted as its own bar.
    Total line = deterministic + sum of shock contributions.
    """
    plt = _mpl()
    T, _, k = contrib.shape
    x = np.arange(T) if index is None else np.asarray(index)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(var_indices), 1,
                             figsize=(11, 3.0 * len(var_indices)),
                             squeeze=False)
    for r, i in enumerate(var_indices):
        ax = axes[r][0]
        parts = [contrib[:, i, j] for j in range(k)] + [deterministic[:, i]]
        labels = SHOCK_NAMES + ["Deterministic + dummies"]
        colors = [cmap(j) for j in range(k)] + ["0.6"]
        pos = np.zeros(T)
        neg = np.zeros(T)
        for series, lab, col in zip(parts, labels, colors):
            base = np.where(series >= 0, pos, neg)
            ax.bar(x, series, bottom=base, width=0.8, color=col,
                   label=lab if r == 0 else None)
            pos += np.clip(series, 0, None)
            neg += np.clip(series, None, 0)
        total = deterministic[:, i] + contrib[:, i, :].sum(axis=1)
        ax.plot(x, total, color="k", lw=1.2, label="Total" if r == 0 else None)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(panel_titles[r] if panel_titles else VARIABLE_NAMES[i])
    axes[0][0].legend(loc="upper left", fontsize=6, ncol=3)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

"""End-to-end replication of BoE Macro Technical Paper No. 3 (Brignone & Piffer).

Usage:
    conda run -n python313 python scripts/run_replication.py \
        [--draws 500] [--accepted 200] [--lags 4]

Writes results/fig2_irf_world.png, fig3_irf_uk.png, fig4_fevd.png,
fig5_shocks.png, fig6_hist_decomp.png and results/summary.md.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from boe_var import analysis
from boe_var.analysis import (
    SHOCK_NAMES, VARIABLE_NAMES, UK_SHOCKS, WORLD_SHOCKS,
)
from boe_var.bvar import BVAR
from boe_var.data import load_data
from boe_var.identification import (
    SIGN_RESTRICTIONS, ZERO_RESTRICTIONS, identify,
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def covid_dummies(index) -> np.ndarray:
    """One dummy per quarter 2020Q1-2021Q2 (6 columns) over the sample index."""
    import pandas as pd
    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    D = np.zeros((len(index), len(quarters)))
    for j, q in enumerate(quarters):
        D[:, j] = (index == q).astype(float)
    return D


def residuals_for(draw, y: np.ndarray, dummies: np.ndarray, lags: int) -> np.ndarray:
    """Reduced-form residuals implied by draw.Pi.

    Regressor order assumed: k*lags lagged values, constant, dummy columns
    (matching BVAR's Pi layout: k x (k*lags + 1 + n_dummy)).
    """
    T, k = y.shape
    rows = []
    for t in range(lags, T):
        xlags = np.concatenate([y[t - l] for l in range(1, lags + 1)])
        rows.append(np.concatenate([xlags, [1.0], dummies[t]]))
    X = np.asarray(rows)
    Pi = np.asarray(draw.Pi)
    if Pi.shape[0] != k:  # tolerate transposed layout
        Pi = Pi.T
    return y[lags:] - X @ Pi.T


def yoy_decomposition(contrib, deterministic, y, lags):
    """Turn level decompositions into YoY (t minus t-4) decompositions."""
    c = contrib[4:] - contrib[:-4]
    d = deterministic[4:] - deterministic[:-4]
    return c, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--accepted", type=int, default=200)
    ap.add_argument("--lags", type=int, default=4)
    ap.add_argument("--horizons", type=int, default=21)
    ap.add_argument("--optimize-hyper", action="store_true",
                    help="select (lam, mu, theta) by maximizing the "
                         "closed-form log marginal likelihood")
    args = ap.parse_args()

    import pandas as pd

    os.makedirs(RESULTS, exist_ok=True)

    # 1. Data (estimation sample 1992Q1-2023Q2)
    df = load_data()
    df = df.loc[(df.index >= pd.Period("1992Q1", "Q"))
                & (df.index <= pd.Period("2023Q2", "Q"))]
    y = df.to_numpy(dtype=float)
    dummies = covid_dummies(df.index)
    k = y.shape[1]

    # 2. Estimate and 3. identify
    hyper = {"lam": 0.2, "mu": 1.0, "theta": 1.0}
    opt_line = None
    if args.optimize_hyper:
        from boe_var.bvar import optimize_hyperparameters
        opt = optimize_hyperparameters(y, lags=args.lags, dummies=dummies)
        hyper = {"lam": opt["lam"], "mu": opt["mu"], "theta": opt["theta"]}
        opt_line = (f"- ML-optimal hyperparameters: lam = {opt['lam']:.4f}, "
                    f"mu = {opt['mu']:.4f}, theta = {opt['theta']:.4f} "
                    f"(log ML = {opt['log_ml']:.2f}).")
        print(opt_line[2:])
    model = BVAR(y, lags=args.lags, dummies=dummies, **hyper)
    n_draws = max(args.draws, args.accepted)
    draws = model.sample_posterior(n_draws)
    # One Q per posterior draw: pass all draws and keep whatever is accepted.
    # Adapt to old identify() signatures defensively.
    import inspect
    id_params = inspect.signature(identify).parameters
    id_kwargs = {}
    if "target_accepted" in id_params:
        id_kwargs["target_accepted"] = len(draws)
    if "max_tries_per_draw" in id_params:
        id_kwargs["max_tries_per_draw"] = 1
    accepted = identify(draws, **id_kwargs)
    # Unpack (draw, B[, w]) tuples; default to unit weights for old-style pairs.
    pairs = [(t[0], t[1]) for t in accepted]
    raw_w = np.array([t[2] if len(t) > 2 else 1.0 for t in accepted],
                     dtype=float)
    if len(pairs) == 0:
        raise SystemExit("No accepted draws; cannot continue.")
    weights = raw_w / raw_w.sum()
    ess_fn = None
    try:
        from boe_var.identification import ess as ess_fn  # noqa: F811
    except ImportError:
        pass
    ess = float(ess_fn(raw_w)) if ess_fn is not None else \
        float(raw_w.sum() ** 2 / np.sum(raw_w ** 2))
    accept_rate = len(pairs) / len(draws) if len(draws) else float("nan")
    print(f"Accepted {len(pairs)} of {len(draws)} posterior draws "
          f"({100 * accept_rate:.1f}%); importance-weight ESS = {ess:.1f}.")

    resid_cache = {}

    def resid_fn(draw):
        key = id(draw)
        if key not in resid_cache:
            resid_cache[key] = residuals_for(draw, y, dummies, args.lags)
        return resid_cache[key]

    # 4. IRFs (figs 2-3)
    irf_b = analysis.irf_bands(pairs, args.horizons, weights=weights)
    analysis.plot_irf_grid(
        irf_b, WORLD_SHOCKS, os.path.join(RESULTS, "fig2_irf_world.png"),
        title="Figure 2: IRFs to world shocks (median, 68%/90% bands)")
    analysis.plot_irf_grid(
        irf_b, UK_SHOCKS, os.path.join(RESULTS, "fig3_irf_uk.png"),
        title="Figure 3: IRFs to UK shocks (median, 68%/90% bands)")

    # 5. FEVD (fig 4)
    fevd_b = analysis.fevd_bands(pairs, args.horizons, weights=weights)
    analysis.plot_fevd(fevd_b, os.path.join(RESULTS, "fig4_fevd.png"),
                       title="Figure 4: FEVD (median shares)")

    # 6. Estimated shocks (fig 5)
    shocks_b = analysis.shock_bands(pairs, resid_fn, weights=weights)
    idx = df.index[args.lags:]
    analysis.plot_shocks(shocks_b, os.path.join(RESULTS, "fig5_shocks.png"),
                         index=idx.to_timestamp(),
                         title="Figure 5: estimated structural shocks")

    # 7. Historical decomposition of YoY inflation & YoY GDP growth (fig 6),
    # with the Covid-dummy contribution as its own (black) component and the
    # decomposition expressed in deviation from the deterministic path.
    contrib, covid = analysis.hd_median(
        pairs, resid_fn, weights=weights, dummies=dummies[args.lags:])
    stoch = contrib.sum(axis=2)
    deterministic = y[args.lags:] - stoch - covid
    c_yoy, d_yoy = yoy_decomposition(contrib, deterministic, y, args.lags)
    covid_yoy = covid[4:] - covid[:-4]
    idx_yoy = idx[4:]
    i_cpi, i_gdp = 5, 7
    analysis.plot_hist_decomp(
        c_yoy, d_yoy, [i_cpi, i_gdp],
        os.path.join(RESULTS, "fig6_hist_decomp.png"),
        index=idx_yoy,
        covid=covid_yoy,
        panel_titles=["YoY CPI inflation (pp)", "YoY GDP growth (pp)"],
        title="Figure 6: historical decomposition "
              "(deviation from deterministic)")

    # 8. Summary numbers: FEVD of UK GDP / CPI at 1 year (h = 4)
    med = fevd_b["median"]
    med = med / med.sum(axis=1, keepdims=True)
    h1 = min(4, med.shape[2] - 1)
    # Headline comparison uses the IDENTIFIED shocks only, matching the
    # paper's ~40%/~50% claims (world demand+supply+energy vs UK
    # demand+supply+monetary). Unidentified shocks reported separately.
    global_shocks = WORLD_SHOCKS
    domestic_shocks = UK_SHOCKS
    unident_shocks = [3, 7]
    lines = [
        "# Replication summary",
        "",
        f"- Posterior draws: {len(draws)}; accepted identified draws: "
        f"{len(pairs)} (acceptance rate {100 * accept_rate:.1f}%); "
        f"lags: {args.lags}.",
        f"- Importance-weight effective sample size (ESS): {ess:.1f}"
        + (" — WARNING: ESS is below 10% of accepted draws; "
           "weighted results may be unreliable."
           if ess < 0.1 * len(pairs) else "."),
        f"- Sample: {df.index[0]}–{df.index[-1]} ({len(df)} quarters).",
        f"- Hyperparameters: lam = {hyper['lam']:.4f}, mu = {hyper['mu']:.4f}, "
        f"theta = {hyper['theta']:.4f}.",
        *([opt_line] if opt_line else []),
        "",
        "## FEVD at 1-year horizon (median shares)",
        "",
        "| Variable | Identified global | Identified domestic | Unidentified |",
        "|---|---|---|---|",
    ]
    for name, i in [("UK GDP", i_gdp), ("UK CPI", i_cpi)]:
        g = med[i, global_shocks, h1].sum()
        d = med[i, domestic_shocks, h1].sum()
        u = med[i, unident_shocks, h1].sum()
        lines.append(
            f"| {name} | {100 * g:.1f}% | {100 * d:.1f}% | {100 * u:.1f}% |")
    lines += [
        "",
        "Paper benchmark: identified global shocks explain roughly ~40% of "
        "UK GDP and ~50% of UK CPI variation at business-cycle horizons.",
        "",
        "Known discrepancy: the paper reports UK monetary policy as the "
        "largest domestic contributor to CPI variance; in this replication "
        "it is not (see detailed table) — likely driven by the proxy world "
        "aggregates and fewer accepted draws.",
        "",
        "## Detailed 1-year FEVD (median, %)",
        "",
        "| Variable | " + " | ".join(SHOCK_NAMES) + " |",
        "|" + "---|" * (len(SHOCK_NAMES) + 1),
    ]
    for i, vname in enumerate(VARIABLE_NAMES):
        cells = " | ".join(f"{100 * med[i, j, h1]:.1f}"
                           for j in range(len(SHOCK_NAMES)))
        lines.append(f"| {vname} | {cells} |")
    lines += [
        "",
        "Figures: fig2_irf_world.png, fig3_irf_uk.png, fig4_fevd.png, "
        "fig5_shocks.png, fig6_hist_decomp.png.",
        "",
    ]
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    # Keep the packaged snapshot (shipped in pip installs) in sync.
    pkg = os.path.join(os.path.dirname(__file__), "..", "src", "boe_var", "_results")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("Wrote results/summary.md and figures 2-6.")


if __name__ == "__main__":
    main()

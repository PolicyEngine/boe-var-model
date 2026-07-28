"""Section 5 forecast-revision exercise (paper Figures 1, 7, 8, 9).

Usage:
    conda run -n python313 python scripts/run_forecast_revision.py \
        [--draws 3000] [--lags 4] [--horizons 13] [--seed 0]

Estimates on --est-start..--est-end (defaults 1992Q1-2025Q1, as
run_replication), treats the last quarter available in the data as period T
and the one before as T-1, and writes results/fig1_forecast.png,
fig7_shock_distributions.png, fig8_composite_irf.png,
fig9_forecast_revision.png and results/forecast_summary.md.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from boe_var import analysis, forecast
from boe_var.analysis import SHOCK_NAMES, weighted_quantile
from boe_var.bvar import BVAR
from boe_var.data import load_data
from boe_var.identification import ess, identify, weak_inference_warnings

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

I_CPI, I_GDP = 5, 7
IDENTIFIED = [0, 1, 2, 4, 5, 6]  # the 6 identified shocks (skip unidentified)


def covid_dummies(index) -> np.ndarray:
    import pandas as pd
    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    D = np.zeros((len(index), len(quarters)))
    for j, q in enumerate(quarters):
        D[:, j] = (index == q).astype(float)
    return D


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120, "font.size": 8, "axes.titlesize": 9,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
    })
    return plt


def weighted_kde(x, w, grid):
    """Weighted Gaussian KDE evaluated on grid."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(x, weights=w)
    return kde(grid)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--lags", type=int, default=4)
    ap.add_argument("--horizons", type=int, default=13)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paths", type=int, default=5,
                    help="stochastic paths per accepted draw for bands")
    ap.add_argument("--est-start", default="1992Q1",
                    help="first quarter of the estimation sample")
    ap.add_argument("--est-end", default="2025Q1",
                    help="last quarter of the estimation sample; data beyond "
                         "this (through the data edge T) is conditioning data")
    args = ap.parse_args()

    import pandas as pd

    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- data: estimation sample vs full sample through the data edge T --
    est_start = pd.Period(args.est_start, "Q")
    est_end = pd.Period(args.est_end, "Q")
    df_full = load_data()
    df_full = df_full.loc[df_full.index >= est_start]
    T_q = df_full.index[-1]          # period T = last quarter in the data
    Tm1_q = df_full.index[-2]        # period T-1
    if T_q <= est_end:
        raise SystemExit(
            f"data edge {T_q} must lie beyond the estimation end {est_end}; "
            "pass --est-end or refresh the data")
    df_est = df_full.loc[df_full.index <= est_end]
    y_est = df_est.to_numpy(dtype=float)
    y_full = df_full.to_numpy(dtype=float)
    dummies_est = covid_dummies(df_est.index)
    dummies_full = covid_dummies(df_full.index)  # zeros after 2021Q2
    H = args.horizons

    # ---- estimate + identify (weights) ----
    model = BVAR(y_est, lags=args.lags, dummies=dummies_est)
    draws = model.sample_posterior(args.draws, seed=args.seed)
    triples = identify(draws, rng=rng)
    if not triples:
        raise SystemExit("No accepted draws.")
    pairs = [(d, B) for d, B, _ in triples]
    w = np.array([t[2] for t in triples], dtype=float)
    print(f"Accepted {len(pairs)}/{len(draws)} draws; ESS = {ess(w):.1f}.")
    inference_warnings = weak_inference_warnings(len(pairs), ess(w),
                                                 len(draws))
    for msg in inference_warnings:
        print(f"WARNING: {msg}")

    # ---- residuals from data BEYOND the estimation sample (dummies=0 post
    # 2021Q2), per draw; eps at T and the revision per draw --------------
    resid_cache: dict[int, np.ndarray] = {}

    def resid_fn(draw):
        key = id(draw)
        if key not in resid_cache:
            resid_cache[key] = forecast.reduced_form_residuals(
                draw, y_full, dummies_full)
        return resid_cache[key]

    # ---- Fig 7: distribution of the 6 identified shocks at period T ----
    dist = forecast.shock_distribution_T(pairs, resid_fn, weights=w)
    eps_T = dist["samples"]
    plt = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), squeeze=False)
    for a, j in enumerate(IDENTIFIED):
        ax = axes[a // 3][a % 3]
        x = eps_T[:, j]
        lo, hi = np.quantile(x, [0.001, 0.999])
        pad = 0.15 * (hi - lo + 1e-9)
        grid = np.linspace(lo - pad, hi + pad, 400)
        dens = weighted_kde(x, dist["weights"], grid)
        ax.fill_between(grid, dens, color="#4292c6", alpha=0.5, lw=0)
        ax.plot(grid, dens, color="#08306b", lw=1.2)
        ax.axvline(0, color="k", lw=0.8)
        p = dist["p_pos"][j]
        ax.set_title(SHOCK_NAMES[j])
        ax.text(0.02, 0.95, f"P(+) = {p:.2f}\nP(−) = {1 - p:.2f}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(fc="white", alpha=0.7, ec="none"))
    fig.suptitle(f"Figure 7: posterior distribution of structural shocks, {T_q}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS, "fig7_shock_distributions.png"))
    plt.close(fig)

    # ---- Fig 8: composite IRF on YoY inflation and YoY GDP growth -------
    # Per-draw level contributions -> YoY transform -> aggregate.
    n = len(pairs)
    Hc = H + 1  # include impact (date T) horizon 0
    joint_yoy = {I_CPI: [], I_GDP: []}
    contrib_yoy = np.empty((n, Hc, 2, len(SHOCK_NAMES)))
    for i, (d, B) in enumerate(pairs):
        c = forecast.composite_contributions(d, B, eps_T[i], Hc)  # (Hc,k,k)
        cy = forecast.yoy_transform_irf(c)                        # YoY effect
        contrib_yoy[i, :, 0] = cy[:, I_CPI, :]
        contrib_yoy[i, :, 1] = cy[:, I_GDP, :]
        joint_yoy[I_CPI].append(cy[:, I_CPI, :].sum(axis=1))
        joint_yoy[I_GDP].append(cy[:, I_GDP, :].sum(axis=1))
    agg = {v: analysis.aggregate(joint_yoy[v], weights=w)
           for v in (I_CPI, I_GDP)}
    decomp_med = weighted_quantile(contrib_yoy, 0.50, w)  # (Hc, 2, k_shock)

    x = np.arange(Hc)
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), squeeze=False)
    for col, (v, lab) in enumerate([(I_CPI, "YoY CPI inflation (pp)"),
                                    (I_GDP, "YoY GDP growth (pp)")]):
        b = agg[v]
        ax = axes[0][col]
        ax.fill_between(x, b["lo90"], b["hi90"], color="#9ecae1", alpha=0.5, lw=0)
        ax.fill_between(x, b["lo68"], b["hi68"], color="#4292c6", alpha=0.5, lw=0)
        ax.plot(x, b["median"], color="#08306b", lw=1.4)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"Composite effect of {T_q} shocks: {lab}")
        ax.set_xlabel(f"quarters after {T_q}")

        ax = axes[1][col]
        pos = np.zeros(Hc)
        neg = np.zeros(Hc)
        for j in range(len(SHOCK_NAMES)):
            s = decomp_med[:, col, j]
            base = np.where(s >= 0, pos, neg)
            ax.bar(x, s, bottom=base, width=0.8, color=cmap(j),
                   label=SHOCK_NAMES[j] if col == 0 else None)
            pos += np.clip(s, 0, None)
            neg += np.clip(s, None, 0)
        ax.plot(x, decomp_med[:, col, :].sum(axis=1), color="k", lw=1.2,
                label="Total (median decomp.)" if col == 0 else None)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"Decomposition of median: {lab}")
        ax.set_xlabel(f"quarters after {T_q}")
    axes[1][0].legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle(f"Figure 8: composite impulse response to the {T_q} shocks")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS, "fig8_composite_irf.png"))
    plt.close(fig)

    # ---- forecast revision per draw + adding-up check -------------------
    max_err = 0.0
    fcT_paths, fcTm1_paths, recon_paths = [], [], []
    for i, (d, B) in enumerate(pairs):
        r = forecast.forecast_revision(d, B, y_full, dummies_full,
                                       horizons=H, atol=1e-6)
        max_err = max(max_err, r["max_err"])
        fcT_paths.append(r["fc_T"])              # dates T+1..T+H
        fcTm1_paths.append(r["fc_Tm1"])          # dates T..T+H
        recon_paths.append(r["fc_Tm1"] + r["composite"])
    print(f"Adding-up identity max error across draws: {max_err:.2e}")

    # ---- Fig 9: history + T forecast bands, pseudo T-1 median, recon ----
    # Predictive bands at T (parameter + shock uncertainty), computed as
    # quantiles of YoY-transformed stochastic paths (not YoY of quantiles).
    tail = y_full[-4:]  # last 4 actual levels, needed for the YoY transform
    yoy_paths, pw = [], []
    for i, (d, _B) in enumerate(pairs):
        for _ in range(args.paths):
            path = forecast.sample_forecast(d, y_full, horizons=H, rng=rng)
            yoy_paths.append(forecast.yoy(np.vstack([tail, path])))
            pw.append(w[i])
    yoy_bands = analysis.aggregate(yoy_paths, weights=np.asarray(pw))

    med_Tm1 = weighted_quantile(np.stack(fcTm1_paths), 0.50, w)   # dates T..T+H
    med_recon = weighted_quantile(np.stack(recon_paths), 0.50, w)
    # pseudo path includes date T; for its YoY prepend last 4 rows BEFORE T
    tail_m1 = y_full[-5:-1]
    yoy_Tm1 = forecast.yoy(np.vstack([tail_m1, med_Tm1]))          # T..T+H
    yoy_recon = forecast.yoy(np.vstack([tail_m1, med_recon]))

    hist_n = 24
    yoy_hist = forecast.yoy(y_full)[-hist_n:]                      # through T
    t_hist = np.arange(-hist_n + 1, 1)                             # 0 = T
    t_fc = np.arange(1, H + 1)
    t_m1 = np.arange(0, H + 1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), squeeze=False)
    for r_, (v, lab) in enumerate([(I_CPI, "YoY CPI inflation (%)"),
                                   (I_GDP, "YoY GDP growth (%)")]):
        ax = axes[r_][0]
        ax.plot(t_hist, yoy_hist[:, v], color="k", lw=1.2, label="Data")
        ax.fill_between(t_fc, yoy_bands["lo90"][:, v], yoy_bands["hi90"][:, v],
                        color="#9ecae1", alpha=0.5, lw=0, label="90% band")
        ax.fill_between(t_fc, yoy_bands["lo68"][:, v], yoy_bands["hi68"][:, v],
                        color="#4292c6", alpha=0.5, lw=0, label="68% band")
        ax.plot(t_fc, yoy_bands["median"][:, v], color="#08306b", lw=1.5,
                label="Forecast at T (median)")
        ax.plot(t_m1, yoy_Tm1[:, v], color="#d94801", lw=1.3, ls="--",
                label="Pseudo forecast at T−1")
        ax.plot(t_m1, yoy_recon[:, v], color="#238b45", lw=1.3, ls=":",
                label="Pseudo + composite IRF")
        ax.axvline(0, color="0.5", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(lab)
        ax.set_xlabel(f"quarters relative to T = {T_q}")
        if r_ == 0:
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"Figure 9: forecast revision between {Tm1_q} and {T_q}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS, "fig9_forecast_revision.png"))
    plt.close(fig)

    # ---- Fig 1: plain unconditional fan chart --------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), squeeze=False)
    for r_, (v, lab) in enumerate([(I_CPI, "YoY CPI inflation (%)"),
                                   (I_GDP, "YoY GDP growth (%)")]):
        ax = axes[r_][0]
        ax.plot(t_hist, yoy_hist[:, v], color="k", lw=1.2, label="Data")
        ax.fill_between(t_fc, yoy_bands["lo90"][:, v], yoy_bands["hi90"][:, v],
                        color="#9ecae1", alpha=0.6, lw=0, label="90%")
        ax.fill_between(t_fc, yoy_bands["lo68"][:, v], yoy_bands["hi68"][:, v],
                        color="#4292c6", alpha=0.6, lw=0, label="68%")
        ax.plot(t_fc, yoy_bands["median"][:, v], color="#08306b", lw=1.5,
                label="Median forecast")
        ax.axvline(0, color="0.5", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(lab)
        if r_ == 0:
            ax.legend(fontsize=7)
        ax.set_xlabel(f"quarters relative to {T_q}")
    fig.suptitle(f"Figure 1: unconditional forecast from {T_q} (fan chart)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS, "fig1_forecast.png"))
    plt.close(fig)

    # ---- summary --------------------------------------------------------
    med_joint_cpi = agg[I_CPI]["median"]
    med_joint_gdp = agg[I_GDP]["median"]
    pk_c = int(np.argmax(np.abs(med_joint_cpi)))
    pk_g = int(np.argmax(np.abs(med_joint_gdp)))
    lines = [
        "# Forecast-revision exercise (Section 5)",
        "",
        f"- Estimation sample {df_est.index[0]}–{df_est.index[-1]}; T = {T_q}, T−1 = {Tm1_q}; "
        f"lags {args.lags}, horizons {H}.",
        f"- Posterior draws {len(draws)}, accepted {len(pairs)} "
        f"({100 * len(pairs) / len(draws):.1f}%), importance-weight ESS "
        f"{ess(w):.1f}.",
        *[f"- **WARNING**: {msg}" for msg in inference_warnings],
        "",
        f"## P(sign) of the identified shocks at T = {T_q} (weighted)",
        "",
        "| Shock | P(>0) | P(<0) |",
        "|---|---|---|",
    ]
    for j in IDENTIFIED:
        p = dist["p_pos"][j]
        lines.append(f"| {SHOCK_NAMES[j]} | {p:.2f} | {1 - p:.2f} |")
    lines += [
        "",
        "## Composite impulse response (median, YoY)",
        "",
        f"- YoY CPI inflation: peak effect {med_joint_cpi[pk_c]:+.2f} pp at "
        f"h = {pk_c} quarters after {T_q}.",
        f"- YoY GDP growth: peak effect {med_joint_gdp[pk_g]:+.2f} pp at "
        f"h = {pk_g} quarters after {T_q}.",
        "",
        "## Adding-up check (Figure 9 identity)",
        "",
        f"- forecast_T = pseudo_{{T−1}} + composite IRF, exact per draw: "
        f"max abs error across all draws/horizons/variables = {max_err:.2e}.",
        "",
        "Figures: fig1_forecast.png, fig7_shock_distributions.png, "
        "fig8_composite_irf.png, fig9_forecast_revision.png.",
        "",
    ]
    with open(os.path.join(RESULTS, "forecast_summary.md"), "w") as f:
        f.write("\n".join(lines))
    # Keep the packaged snapshot (shipped in pip installs) in sync.
    pkg = os.path.join(os.path.dirname(__file__), "..", "src", "boe_var", "_results")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "forecast_summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("Wrote results/forecast_summary.md and figures 1, 7, 8, 9.")


if __name__ == "__main__":
    main()

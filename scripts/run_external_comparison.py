#!/usr/bin/env python
"""Regenerate the SVAR-side numbers quoted in results/forecast_vs_boe_obr.md.

The comparison document sets this model's forecast against the Bank of
England's Monetary Policy Report central projection and the OBR's Economic
and fiscal outlook. Those two comparators are published numbers, transcribed
in the document with their source table. Everything on *our* side is produced
here, so no number in that document is unreproducible.

Prints, for UK CPI and UK GDP (four-quarter rates, the basis both comparators
publish):

1. the weighted-median forecast path and 68% band from the production sample;
2. the same path re-estimated on a pre-Covid sample, which is what identifies
   the estimation window (not the conditioning rate path) as the source of the
   medium-term inflation gap;
3. the identified UK monetary-policy shock's IRF, used to size how much of the
   gap the rate-path difference can account for.

Usage:
    conda run -n python313 python scripts/run_external_comparison.py \
        [--draws 6000] [--lags 4] [--horizons 14] [--paths 5] [--seed 20260812]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from boe_var import analysis, forecast
from boe_var.bvar import BVAR
from boe_var.data import load_data
from boe_var.identification import ess, identify

I_BANK_RATE, I_CPI, I_GDP = 3, 5, 7
I_MP_SHOCK = 6  # 'UK mon. pol.' in analysis.SHOCK_NAMES

# The quarters the BoE MPR and the OBR EFO both publish on.
COMPARISON_QUARTERS = ("2026Q3", "2027Q3", "2028Q3", "2029Q3")


def covid_dummies(index) -> np.ndarray:
    """One dummy per pandemic quarter, 2020Q1-2021Q2 (see docs/methodology.md)."""
    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    D = np.zeros((len(index), len(quarters)))
    for j, q in enumerate(quarters):
        D[:, j] = (index == q).astype(float)
    return D


def fit(df_est, draws: int, lags: int, seed: int, rng):
    model = BVAR(df_est.to_numpy(dtype=float), lags=lags,
                 dummies=covid_dummies(df_est.index))
    triples = identify(model.sample_posterior(draws, seed=seed), rng=rng)
    if not triples:
        raise SystemExit("No accepted draws.")
    pairs = [(d, B) for d, B, _ in triples]
    w = np.array([t[2] for t in triples], dtype=float)
    return pairs, w


def yoy_forecast(pairs, w, y_full, horizons: int, n_paths: int, rng):
    """Weighted quantiles of the stochastic forecast paths, YoY and in levels.

    Quantiles are taken of the YoY paths, not YoY of the quantiles -- the same
    convention as run_forecast_revision.py. Both are returned because
    ``forecast.yoy`` is only meaningful for the 100*log variables: Bank Rate
    enters the VAR as a level (per cent), so its YoY transform is a
    four-quarter *change* in the rate, not a growth rate. Read Bank Rate off
    the level aggregate.
    """
    tail = y_full[-4:]  # actual levels needed for the first four YoY differences
    yoy_paths, level_paths, weights = [], [], []
    for i, (draw, _B) in enumerate(pairs):
        for _ in range(n_paths):
            path = forecast.sample_forecast(draw, y_full, horizons=horizons, rng=rng)
            yoy_paths.append(forecast.yoy(np.vstack([tail, path])))
            level_paths.append(path)
            weights.append(w[i])
    weights = np.asarray(weights)
    return (analysis.aggregate(yoy_paths, weights=weights),
            analysis.aggregate(level_paths, weights=weights))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=int, default=6000)
    p.add_argument("--lags", type=int, default=4)
    p.add_argument("--horizons", type=int, default=14)
    p.add_argument("--paths", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--est-start", default="1992Q1")
    p.add_argument("--est-end", default="2025Q1")
    p.add_argument("--precovid-end", default="2019Q4")
    args = p.parse_args()

    df = load_data()
    df = df.loc[df.index >= pd.Period(args.est_start, "Q")]
    y_full = df.to_numpy(dtype=float)
    origin = df.index[-1]
    quarters = pd.period_range(origin + 1, periods=args.horizons, freq="Q")
    at = {str(q): h for h, q in enumerate(quarters)}
    wanted = [q for q in COMPARISON_QUARTERS if q in at]
    if not wanted:
        raise SystemExit(
            f"forecast origin {origin} reaches {quarters[-1]}, which covers none of "
            f"{COMPARISON_QUARTERS}; pass a longer --horizons")

    print(f"Forecast origin T = {origin} (last quarter in the data).")
    print(f"Sample {df.index[0]}-{args.est_end}; lags {args.lags}; "
          f"{args.draws} draws; {args.paths} paths per draw.\n")

    def report(est_end: str, label: str):
        rng = np.random.default_rng(args.seed)
        df_est = df.loc[df.index <= pd.Period(est_end, "Q")]
        pairs, w = fit(df_est, args.draws, args.lags, args.seed, rng)
        print(f"--- {label} (estimated to {est_end}): "
              f"{len(pairs)} accepted draws, ESS {ess(w):.1f}")
        agg, levels = yoy_forecast(pairs, w, y_full, args.horizons,
                                   args.paths, rng)
        print(f"{'Quarter':<8} {'CPI med':>8} {'CPI 68%':>16} "
              f"{'GDP med':>8} {'GDP 68%':>16} {'Bank Rate':>10}")
        print(f"{'':8} {'YoY %':>8} {'':>16} {'YoY %':>8} {'':>16} {'level, %':>10}")
        for q in wanted:
            h = at[q]
            print(f"{q:<8} {agg['median'][h, I_CPI]:>8.2f} "
                  f"{agg['lo68'][h, I_CPI]:>7.2f}-{agg['hi68'][h, I_CPI]:<8.2f} "
                  f"{agg['median'][h, I_GDP]:>8.2f} "
                  f"{agg['lo68'][h, I_GDP]:>7.2f}-{agg['hi68'][h, I_GDP]:<8.2f} "
                  f"{levels['median'][h, I_BANK_RATE]:>10.2f}")
        print()
        return pairs, w

    pairs, w = report(args.est_end, "Production sample")
    report(args.precovid_end, "Pre-Covid sample")

    # How much of the CPI gap can the rate-path difference explain? Scale the
    # identified MP-shock IRF to the gap between our Bank Rate path and the
    # market curve the MPR conditions on. Partial equilibrium: this ignores
    # expectations/anchoring channels the SVAR cannot represent.
    med = analysis.irf_bands(pairs, horizons=args.horizons + 2, weights=w)["median"]
    impact = med[I_BANK_RATE, I_MP_SHOCK, 0]
    level = med[I_CPI, I_MP_SHOCK, :]
    yoy_effect = level.copy()
    yoy_effect[4:] = level[4:] - level[:-4]
    print("--- Identified UK monetary-policy shock (median IRF)")
    print(f"Bank Rate on impact: {impact:+.3f}pp")
    print(f"YoY CPI effect at h=8..13: "
          f"{np.round(yoy_effect[8:14], 3)}")
    for bp in (0.35,):
        scaled = yoy_effect[8:14] * bp / abs(impact)
        print(f"Scaled to a {bp * 100:.0f}bp rate-path shortfall: "
              f"{scaled.min():+.2f} to {scaled.max():+.2f}pp on YoY CPI")

    hist = forecast.yoy(df.loc[df.index <= pd.Period(args.est_end, "Q")]
                        .to_numpy(dtype=float))
    print(f"\nHistorical mean YoY over the estimation sample: "
          f"CPI {hist[:, I_CPI].mean():.2f}%, GDP {hist[:, I_GDP].mean():.2f}% "
          "(for contrast with the forecast plateau -- they are not the same thing).")


if __name__ == "__main__":
    main()

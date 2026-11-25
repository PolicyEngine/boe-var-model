"""Held-out forecast accuracy and band coverage at different Minnesota
tightness values (issues #11 and #12).

    uv run python scripts/run_lambda_sensitivity.py

Writes results/lambda_sensitivity.json and results/lambda_sensitivity.md.

Why: the published model fixes the Minnesota tightness at lam = 0.2. The
marginal likelihood prefers about 1.04, and at that value the impulse
responses sit much closer to the paper's Figures 2-3 (issue #11). Before
anyone moves the published prior, this measures what the change would do to
the forecasts: the same expanding-window evaluations that produce
results/coverage_evaluation.json and results/rolling_evaluation.json, on the
published specification (six Covid dummies), with lam as the only change.

Nothing here changes the published model. The evaluation code is not
edited: ``run_coverage_evaluation.run_coverage`` builds ``BVAR(train, ...)``
with the default lam, so this script swaps that one name for a BVAR with
lam fixed (see ``coverage_at``). At lam = 0.2 the result must equal the
committed coverage file cell for cell, which the tests check.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import run_coverage_evaluation as cov  # noqa: E402

from boe_var.bvar import BVAR  # noqa: E402
from boe_var.data import COLUMNS, load_data  # noqa: E402
from boe_var.evaluation import (  # noqa: E402
    evaluation_code_version,
    rolling_origin_evaluation,
)

OUT_JSON = os.path.join(ROOT, "results", "lambda_sensitivity.json")
OUT_MD = os.path.join(ROOT, "results", "lambda_sensitivity.md")

LAMBDAS = (0.2, 0.5, 1.04)      # published; midway; marginal-likelihood optimum
BASE = 0.2
HEADLINE = ("uk_gdp", "cpisa")
LABELS = {"uk_gdp": "UK GDP", "cpisa": "UK CPI"}


def code_version() -> str:
    return evaluation_code_version(extra_paths=(
        os.path.abspath(__file__),
        os.path.join(HERE, "run_coverage_evaluation.py"),
    ))


def covid_dummies(index) -> np.ndarray:
    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    return np.column_stack([(index == q).astype(float) for q in quarters])


def coverage_at(lam: float, y, dummies, *, n_draws: int, first_origin: int,
                horizons: int = cov.HORIZONS) -> dict:
    """The committed coverage evaluation with ``lam`` as the only change."""
    record, yoy = {}, {}
    original = cov.BVAR
    cov.BVAR = functools.partial(BVAR, lam=lam)
    try:
        hits, n = cov.run_coverage(
            y, lags=cov.LAGS, horizons=horizons, first_origin=first_origin,
            n_draws=n_draws, n_paths=cov.N_PATHS, seed=cov.SEED,
            dummies=dummies, record=record, yoy=yoy)
    finally:
        cov.BVAR = original
    taus = list(cov.TAUS)
    Q = record["quantiles"]                       # (origins, h, k, taus)
    width90 = (Q[..., taus.index(0.95)] - Q[..., taus.index(0.05)]).mean(axis=0)
    width68 = (Q[..., taus.index(0.84)] - Q[..., taus.index(0.16)]).mean(axis=0)
    crps = cov.quantile_crps(Q, record["actual"]).mean(axis=0)   # (h, k)
    return {
        "coverage": cov.coverage_block(hits, n, horizons),
        "coverage_yoy": cov.coverage_block(yoy["hits"], n, horizons),
        "width90": width90, "width68": width68, "crps": crps,
        "origins": int(n[0]),
    }


def _mean_cov(block: dict, level: str) -> float:
    return float(np.mean([v for h in block[level].values()
                          for v in h.values()]))


def _by_var(arr) -> dict:
    """(h, k) array -> {variable: mean over horizons}."""
    return {c: float(arr[:, j].mean()) for j, c in enumerate(COLUMNS)}


def build_report(lambdas=LAMBDAS, n_draws: int = cov.N_DRAWS,
                 first_origin: int = cov.FIRST_ORIGIN,
                 horizons: int = cov.HORIZONS) -> dict:
    df = load_data()
    y = df.to_numpy(float)
    dummies = covid_dummies(df.index)
    runs, raw = {}, {}
    for lam in lambdas:
        c = raw[lam] = coverage_at(lam, y, dummies, n_draws=n_draws,
                                   first_origin=first_origin,
                                   horizons=horizons)
        roll = rolling_origin_evaluation(
            y, lags=cov.LAGS, horizons=horizons, first_origin=first_origin,
            dummies=dummies, bvar_kwargs={"lam": lam})
        rel = {f"h{row['horizon']}": dict(zip(COLUMNS, map(float, row["relative_rmse"])))
               for row in roll["horizons"]}
        rmse = {f"h{row['horizon']}": dict(zip(COLUMNS, map(float, row["bvar_rmse"])))
                for row in roll["horizons"]}
        runs[f"{lam:g}"] = {
            "lam": lam,
            "coverage": c["coverage"],
            "coverage_yoy": c["coverage_yoy"],
            "mean_coverage": {
                "levels": {lv: round(_mean_cov(c["coverage"], lv), 4)
                           for lv in ("68", "90")},
                "yoy": {lv: round(_mean_cov(c["coverage_yoy"], lv), 4)
                        for lv in ("68", "90")},
            },
            "relative_rmse_vs_random_walk": rel,
            "point_rmse": rmse,
        }
    base = raw[BASE] if BASE in raw else raw[lambdas[0]]
    base_key = f"{(BASE if BASE in raw else lambdas[0]):g}"
    for lam in lambdas:
        c, r = raw[lam], runs[f"{lam:g}"]
        r["width90_ratio_to_base"] = {
            k: round(v, 4) for k, v in _by_var(c["width90"] / base["width90"]).items()}
        r["crps_ratio_to_base"] = {
            k: round(v, 4) for k, v in _by_var(c["crps"] / base["crps"]).items()}
        r["point_rmse_ratio_to_base"] = {
            h: {v: round(r["point_rmse"][h][v] / runs[base_key]["point_rmse"][h][v], 4)
                for v in COLUMNS}
            for h in r["point_rmse"]}
    return {
        "method": "the committed coverage and rolling-origin evaluations on "
                  "the published specification (six Covid dummies), re-run "
                  "with the Minnesota tightness lam as the only change",
        "note": "Report only. The published model uses lam = 0.2. Coverage "
                "is of raw model bands, before any band_scale_factors. "
                "width90, crps and point_rmse ratios are to the lam = "
                f"{base_key} run, averaged over horizons h1-h{horizons} "
                "(below 1 = narrower / better).",
        "lambdas": list(lambdas),
        "base": base_key,
        "draws_per_origin": n_draws,
        "paths_per_draw": cov.N_PATHS,
        "seed": cov.SEED,
        "lags": cov.LAGS,
        "horizons": horizons,
        "first_origin": first_origin,
        "origins": base["origins"],
        "sample": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "covid_dummies": "2020Q1-2021Q2",
        "code_version": code_version(),
        "runs": runs,
    }


def markdown(report: dict) -> str:
    runs = report["runs"]
    keys = list(runs)
    H = report["horizons"]
    L = [
        "# Forecast accuracy and band coverage at different prior tightness",
        "",
        "Generated by `scripts/run_lambda_sensitivity.py`; do not edit by "
        "hand. Issues #11 and #12.",
        "",
        f"- Published specification (six Covid dummies), {report['origins']} "
        f"expanding-window origins, h1–h{H}, "
        f"{report['draws_per_origin']:,} posterior draws × "
        f"{report['paths_per_draw']} paths per origin, seed {report['seed']}, "
        f"data {report['sample']['start']}–{report['sample']['end']}.",
        "- The Minnesota tightness λ is the only thing that changes. The "
        "published model uses λ = 0.2; 1.04 is the marginal-likelihood "
        "optimum.",
        "- Coverage is of the raw model bands, before any calibration factor. "
        "With 50 overlapping origins the standard error of a coverage figure "
        "is about 0.04 at 90% and 0.07 at 68% for a single cell.",
        "",
        "## All eight variables, mean over horizons",
        "",
        "| λ | 68% coverage, levels | 90% coverage, levels | 68% coverage, "
        "year on year | 90% coverage, year on year | 90% band width / λ 0.2 | "
        "CRPS / λ 0.2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in keys:
        r = runs[k]
        mc = r["mean_coverage"]
        w = np.mean(list(r["width90_ratio_to_base"].values()))
        c = np.mean(list(r["crps_ratio_to_base"].values()))
        L.append(f"| {k} | {mc['levels']['68']:.3f} | {mc['levels']['90']:.3f} | "
                 f"{mc['yoy']['68']:.3f} | {mc['yoy']['90']:.3f} | {w:.2f} | "
                 f"{c:.3f} |")
    for var in HEADLINE:
        L += ["", f"## {LABELS[var]}", "",
              "Year-on-year band coverage (the basis of the published fan) "
              "and point-forecast RMSE relative to a random walk.",
              "",
              "| λ | 90% cov. h1 | h4 | h8 | 68% cov. h1 | h4 | h8 | "
              "RMSE / RW h1 | h4 | h8 | 90% width / λ 0.2 | CRPS / λ 0.2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in keys:
            r = runs[k]
            hs = [h for h in (1, 4, 8) if h <= H]
            cells = [f"{r['coverage_yoy'][lv][f'h{h}'][var]:.2f}"
                     for lv in ("90", "68") for h in hs]
            cells += [f"{r['relative_rmse_vs_random_walk'][f'h{h}'][var]:.2f}"
                      for h in hs]
            cells += [f"{r['width90_ratio_to_base'][var]:.2f}",
                      f"{r['crps_ratio_to_base'][var]:.3f}"]
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "## CRPS relative to λ = 0.2, by variable (mean over horizons)",
          "", "| λ | " + " | ".join(COLUMNS) + " |",
          "|---|" + "---|" * len(COLUMNS)]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(
            f"{runs[k]['crps_ratio_to_base'][v]:.3f}" for v in COLUMNS) + " |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=cov.N_DRAWS)
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-md", default=OUT_MD)
    args = ap.parse_args()
    report = build_report(n_draws=args.draws)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    with open(args.out_md, "w") as f:
        f.write(markdown(report))
    for k, r in report["runs"].items():
        print(k, r["mean_coverage"])


if __name__ == "__main__":
    main()

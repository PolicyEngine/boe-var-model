"""Pin this replication's impulse responses and set them against the paper's
Figures 2-3 (issue #11).

    uv run python scripts/run_irf_reference.py

Writes results/irf_reference.json and results/irf_vs_paper.md.

The run is the published replication (`scripts/run_replication.py --draws
10000 --accepted 10000`): 1992Q1-2025Q1, six Covid dummies, lam 0.2 / mu 1 /
theta 1, seed 20260812, one rotation per posterior draw, importance-weighted
quantiles. `irf_reference.json` holds the pointwise median and 68% / 90%
sets for the six identified shocks at h = 0..20, in the paper's units
(percent; basis points for Bank Rate). Responses are to one-standard-deviation
shocks on the level of each variable, as in the paper (section 3.1).

The paper side is data/external/brignone_piffer_2025_fig2_fig3_irf_medians.csv,
read from the vector drawing of the figures by
`scripts/extract_paper_irf_medians.py`. The comparison is a report. The
paper's world series are unpublished and proxied here, its sample ends
earlier and its data vintage is older, so the two sets of numbers are not
expected to agree closely; the report says where they do and do not.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from boe_var import analysis  # noqa: E402
from boe_var.bvar import BVAR  # noqa: E402
from boe_var.data import load_data  # noqa: E402
from boe_var.evaluation import evaluation_code_version  # noqa: E402
from boe_var.identification import (  # noqa: E402
    SHOCKS,
    SIGN_RESTRICTIONS,
    VARIABLES,
    ess,
    identify,
)

PAPER_CSV = os.path.join(ROOT, "data", "external",
                         "brignone_piffer_2025_fig2_fig3_irf_medians.csv")
OUT_JSON = os.path.join(ROOT, "results", "irf_reference.json")
OUT_MD = os.path.join(ROOT, "results", "irf_vs_paper.md")

DRAWS = 10_000
SEED = 20260812
LAGS = 4
HORIZONS = 21
EST_START, EST_END = "1992Q1", "2025Q1"
HYPER = {"lam": 0.2, "mu": 1.0, "theta": 1.0}
IDENTIFIED = ["world_demand", "world_supply", "world_energy",
              "uk_demand", "uk_supply", "uk_monpol"]
# model units -> paper units (Bank Rate: percentage points -> basis points)
UNIT_SCALE = {v: 1.0 for v in VARIABLES} | {"bank_rate": 100.0}
REPORT_HORIZONS = (0, 4, 8, 20)
# The file stores six decimals; two operating systems agreed to 3e-6.
CHECK_TOL = 1e-4
LABELS = {
    "world_gdp": "World GDP", "world_cpi": "World CPI", "oil_price": "Oil price",
    "bank_rate": "Bank Rate (bp)", "eri": "Exchange rate", "cpisa": "UK CPI",
    "cpi_energy": "CPI energy", "uk_gdp": "UK GDP",
    "world_demand": "World demand", "world_supply": "World supply",
    "world_energy": "World energy", "uk_demand": "UK demand",
    "uk_supply": "UK supply", "uk_monpol": "UK monetary policy",
}


def code_version() -> str:
    src = os.path.join(ROOT, "src", "boe_var")
    return evaluation_code_version(extra_paths=(
        os.path.join(src, "analysis.py"),
        os.path.join(src, "identification.py"),
        os.path.abspath(__file__),
        PAPER_CSV,
    ))


def covid_dummies(index) -> np.ndarray:
    import pandas as pd

    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    return np.column_stack([(index == q).astype(float) for q in quarters])


def reference_irfs(n_draws: int = DRAWS, seed: int = SEED,
                   horizons: int = HORIZONS, est_end: str = EST_END,
                   lam: float = HYPER["lam"], lags: int = LAGS) -> dict:
    """Weighted median and 68/90% sets, in the paper's units.

    Returns {"accepted", "ess", "irf": {shock: {variable: {stat: [h]}}}}.
    """
    import pandas as pd

    df = load_data()
    df = df.loc[(df.index >= pd.Period(EST_START, "Q"))
                & (df.index <= pd.Period(est_end, "Q"))]
    y = df.to_numpy(dtype=float)
    model = BVAR(y, lags=lags, dummies=covid_dummies(df.index),
                 **(HYPER | {"lam": lam}))
    draws = model.sample_posterior(n_draws, seed=seed)
    accepted = identify(draws, rng=np.random.default_rng(seed + 1),
                        target_accepted=len(draws))
    if not accepted:
        raise SystemExit("no accepted draws")
    pairs = [(d, B) for d, B, _ in accepted]
    w = np.array([t[2] for t in accepted], dtype=float)
    bands = analysis.irf_bands(pairs, horizons, weights=w / w.sum())
    out = {}
    for shock in IDENTIFIED:
        j = SHOCKS.index(shock)
        out[shock] = {}
        for i, var in enumerate(VARIABLES):
            out[shock][var] = {
                stat: [round(float(v) * UNIT_SCALE[var], 6)
                       for v in bands[stat][i, j]]
                for stat in ("median", "lo68", "hi68", "lo90", "hi90")
            }
    return {"accepted": len(pairs), "ess": round(float(ess(w)), 1),
            "irf": out}


def load_paper(path: str = PAPER_CSV) -> dict:
    """{shock: {variable: {"median": [h], "resolution": float}}}."""
    out: dict = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            cell = out.setdefault(r["shock"], {}).setdefault(
                r["variable"], {"median": [None] * HORIZONS,
                                "resolution": float(r["resolution"])})
            cell["median"][int(r["horizon"])] = float(r["median"])
    return out


def compare(ours: dict, paper: dict) -> dict:
    """Where the paper's median sits relative to this replication's sets."""
    cells = []
    for shock in IDENTIFIED:
        for var in VARIABLES:
            o, p = ours[shock][var], paper[shock][var]
            pm = np.asarray(p["median"])
            om = np.asarray(o["median"])
            in68 = (np.asarray(o["lo68"]) <= pm) & (pm <= np.asarray(o["hi68"]))
            in90 = (np.asarray(o["lo90"]) <= pm) & (pm <= np.asarray(o["hi90"]))
            # sign is only compared where the paper's value is clear of zero
            # by more than three times the reading resolution
            clear = np.abs(pm) > 3.0 * p["resolution"]
            cells.append({
                "shock": shock, "variable": var,
                "inside_68": int(in68.sum()), "inside_90": int(in90.sum()),
                "sign_compared": int(clear.sum()),
                "sign_agrees": int((np.sign(pm) == np.sign(om))[clear].sum()),
                # distance between the medians, as a share of the paper's
                # largest response in the panel: unlike the "inside" counts
                # it does not improve when this replication's sets widen
                "scaled_gap": round(float(np.abs(om - pm).mean()
                                          / np.abs(pm).max()), 4),
                "paper": {f"h{h}": round(float(pm[h]), 4)
                          for h in REPORT_HORIZONS},
                "ours": {f"h{h}": round(float(om[h]), 4)
                         for h in REPORT_HORIZONS},
            })
    n = len(cells) * HORIZONS
    compared = sum(c["sign_compared"] for c in cells)
    # impact responses the paper restricts in sign: both medians are clear
    # of zero there, so their ratio is well defined
    ratios = {}
    for shock in IDENTIFIED:
        for i, _ in SIGN_RESTRICTIONS[shock]:
            var = VARIABLES[i]
            ratios[f"{shock}:{var}"] = round(
                ours[shock][var]["median"][0] / paper[shock][var]["median"][0],
                3)
    return {
        "impact_ratio_sign_restricted": {
            "note": "this replication's median impact response divided by "
                    "the paper's, for the 23 responses Table 2 restricts in "
                    "sign",
            "median": round(float(np.median(list(ratios.values()))), 3),
            "min": min(ratios.values()),
            "max": max(ratios.values()),
            "by_response": ratios,
        },
        "cells": cells,
        "points": n,
        "share_inside_68": round(sum(c["inside_68"] for c in cells) / n, 4),
        "share_inside_90": round(sum(c["inside_90"] for c in cells) / n, 4),
        "sign_points_compared": compared,
        "share_sign_agrees": round(
            sum(c["sign_agrees"] for c in cells) / compared, 4),
        "panels_fully_inside_90": sum(c["inside_90"] == HORIZONS
                                      for c in cells),
        "panels_fully_outside_90": sum(c["inside_90"] == 0 for c in cells),
        "median_scaled_gap": round(float(np.median(
            [c["scaled_gap"] for c in cells])), 4),
        "mean_scaled_gap": round(float(np.mean(
            [c["scaled_gap"] for c in cells])), 4),
    }


def _fmt(x: float) -> str:
    return f"{x:.2f}" if abs(x) < 10 else f"{x:.1f}"


def markdown(report: dict) -> str:
    cmp_ = report["paper_comparison"]
    L = [
        "# Impulse responses: this replication against the paper's Figures 2-3",
        "",
        "Generated by `scripts/run_irf_reference.py`; do not edit by hand. "
        "Issue #11.",
        "",
        "- **This replication:** the published run "
        f"({report['draws']:,} posterior draws, seed {report['seed']}, "
        f"{report['sample']['start']}–{report['sample']['end']}, "
        f"{report['lags']} lags, lam {report['hyperparameters']['lam']:g}, "
        f"{report['accepted']} accepted, ESS {report['ess']}), "
        "importance-weighted pointwise medians and 68% / 90% sets "
        "(`results/irf_reference.json`).",
        "- **Paper:** medians of Brignone & Piffer (2025) Figures 2–3, read "
        "from the vector drawing in the PDF "
        "(`data/external/brignone_piffer_2025_fig2_fig3_irf_medians.csv`; "
        "method and checks in `data/external/README.md`). They are not "
        "published as numbers and the authors have not confirmed them.",
        "- Responses are to a one-standard-deviation shock, on the level of "
        "each variable, in percent (Bank Rate in basis points), 0–20 quarters.",
        "- The paper's world GDP and world CPI are unpublished Bank series "
        "(proxied here), its sample ends earlier, and it uses 10,000 accepted "
        "draws. Differences below are not all replication errors.",
        "",
        "## Summary",
        "",
        f"Across {cmp_['points']} points (48 panels × 21 quarters):",
        "",
        f"- the paper's median lies inside this replication's 68% set at "
        f"{100 * cmp_['share_inside_68']:.0f}% of points and inside the 90% "
        f"set at {100 * cmp_['share_inside_90']:.0f}%;",
        f"- {cmp_['panels_fully_inside_90']} of 48 panels have the paper's "
        "median inside the 90% set at every quarter, and "
        f"{cmp_['panels_fully_outside_90']} have it outside at every quarter;",
        f"- where the paper's median is clear of zero "
        f"({cmp_['sign_points_compared']} points), the two medians have the "
        f"same sign at {100 * cmp_['share_sign_agrees']:.0f}%;",
        "- the average distance between the two medians over the 21 "
        "quarters, as a share of the paper's largest response in the panel, "
        f"is {100 * cmp_['median_scaled_gap']:.0f}% for the median panel "
        f"and {100 * cmp_['mean_scaled_gap']:.0f}% on average.",
        "",
        "A 90% set is not a test of equality with another study's median: "
        "it measures this replication's own posterior uncertainty. The "
        "shares say how far apart the two are relative to that uncertainty, "
        "and they rise when the sets widen. The distance in the last line "
        "does not.",
        "",
    ]
    imp = cmp_["impact_ratio_sign_restricted"]
    big = max(imp["by_response"], key=imp["by_response"].get)
    rest = sorted(v for k, v in imp["by_response"].items() if k != big)
    L += [
        "## Impact responses",
        "",
        "For the 23 impact responses that Table 2 restricts in sign, this "
        "replication's median divided by the paper's has a median of "
        f"{imp['median']:.2f}. Twenty-two of them lie between {rest[0]:.2f} "
        f"and {rest[-1]:.2f}; the other is "
        f"{LABELS[big.split(':')[0]].lower()} → {LABELS[big.split(':')[1]]} "
        f"at {imp['by_response'][big]:.2f}.",
        "",
        "## Panels where the paper's median leaves the 90% set",
        "",
        "| Shock | Variable | Quarters inside 68% | Quarters inside 90% | "
        "Same sign at h = 20 |",
        "|---|---|---|---|---|",
    ]
    for c in sorted(cmp_["cells"], key=lambda c: c["inside_90"]):
        if c["inside_90"] < HORIZONS:
            same = (c["paper"]["h20"] > 0) == (c["ours"]["h20"] > 0)
            L.append(f"| {LABELS[c['shock']]} | {LABELS[c['variable']]} | "
                     f"{c['inside_68']} | {c['inside_90']} | "
                     f"{'yes' if same else 'no'} |")
    L.append("")
    by = {(c["shock"], c["variable"]): c for c in cmp_["cells"]}
    for shock in IDENTIFIED:
        L += [f"## {LABELS[shock]} shock", "",
              "Median response, paper / this replication. "
              "\"In 90%\" counts the quarters (of 21) at which the paper's "
              "median is inside this replication's 90% set.",
              "",
              "| Variable | h = 0 | h = 4 | h = 8 | h = 20 | In 68% | In 90% |",
              "|---|---|---|---|---|---|---|"]
        for var in VARIABLES:
            c = by[(shock, var)]
            cols = [f"{_fmt(c['paper'][f'h{h}'])} / {_fmt(c['ours'][f'h{h}'])}"
                    for h in REPORT_HORIZONS]
            L.append(f"| {LABELS[var]} | " + " | ".join(cols)
                     + f" | {c['inside_68']} | {c['inside_90']} |")
        L.append("")
    return "\n".join(L)


def build_report(n_draws: int = DRAWS, seed: int = SEED,
                 est_end: str = EST_END, lam: float = HYPER["lam"],
                 lags: int = LAGS) -> dict:
    ref = reference_irfs(n_draws, seed, est_end=est_end, lam=lam, lags=lags)
    return {
        "method": "importance-weighted pointwise median and 68/90% credible "
                  "sets of the impulse responses to one-standard-deviation "
                  "identified shocks, on levels, in the paper's units "
                  "(percent; basis points for bank_rate)",
        "draws": n_draws,
        "seed": seed,
        "lags": lags,
        "hyperparameters": HYPER | {"lam": lam},
        "covid_dummies": "2020Q1-2021Q2",
        "sample": {"start": EST_START, "end": est_end},
        "accepted": ref["accepted"],
        "ess": ref["ess"],
        "horizons": list(range(HORIZONS)),
        "variables": VARIABLES,
        "shocks": IDENTIFIED,
        "code_version": code_version(),
        "irf": ref["irf"],
        "paper_comparison": compare(ref["irf"], load_paper()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--est-end", default=EST_END,
                    help="last quarter of the estimation sample. The "
                         "committed file uses 2025Q1; 2023Q2 is the paper's "
                         "own sample end, for a sensitivity run written "
                         "elsewhere with --out-json / --out-md")
    ap.add_argument("--lam", type=float, default=HYPER["lam"],
                    help="Minnesota tightness, for a sensitivity run (the "
                         "committed file uses 0.2; the marginal-likelihood "
                         "optimum is about 1.04)")
    ap.add_argument("--lags", type=int, default=LAGS,
                    help="lag length, for a sensitivity run (the paper does "
                         "not state it; the committed file uses 4)")
    ap.add_argument("--out-json", default=OUT_JSON)
    ap.add_argument("--out-md", default=OUT_MD)
    ap.add_argument("--check", action="store_true",
                    help="after writing, compare every number with the "
                         "committed results/irf_reference.json and exit 1 "
                         "on any difference above 1e-4 (CI, manual)")
    args = ap.parse_args()
    report = build_report(args.draws, args.seed, args.est_end, args.lam,
                          args.lags)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    with open(args.out_md, "w") as f:
        f.write(markdown(report))
    c = report["paper_comparison"]
    print(f"accepted {report['accepted']}, ESS {report['ess']}; paper median "
          f"inside 68% set {c['share_inside_68']:.3f}, 90% set "
          f"{c['share_inside_90']:.3f}; sign agreement "
          f"{c['share_sign_agrees']:.3f}")
    if args.check:
        with open(OUT_JSON) as f:
            committed = json.load(f)
        worst = max_difference(committed, report)
        print(f"largest difference from the committed file: {worst:.3g}")
        if worst > CHECK_TOL:
            raise SystemExit("irf_reference.json does not reproduce")


def max_difference(a: dict, b: dict) -> float:
    """Largest absolute difference between two reports' impulse responses;
    infinite if the accepted count or the code version differs."""
    if (a["accepted"], a["code_version"]) != (b["accepted"], b["code_version"]):
        return float("inf")
    worst = 0.0
    for shock in IDENTIFIED:
        for var in VARIABLES:
            for stat, vals in a["irf"][shock][var].items():
                diff = np.abs(np.asarray(vals)
                              - np.asarray(b["irf"][shock][var][stat]))
                worst = max(worst, float(diff.max()))
    return worst


if __name__ == "__main__":
    main()

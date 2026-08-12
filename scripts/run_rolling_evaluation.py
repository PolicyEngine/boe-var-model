"""Generate leakage-safe rolling forecast evidence for the packaged dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from boe_var.data import COLUMNS, load_data
from boe_var.evaluation import (
    benjamini_hochberg,
    evaluation_code_version,
    rolling_origin_evaluation,
)

# The Diebold-Mariano grid is 8 variables x 8 horizons per benchmark. Reading
# an unadjusted p = 0.02 out of 64 tries as evidence of skill is a
# multiple-comparisons error, so every p-value block is accompanied by
# Benjamini-Hochberg q-values computed over the whole grid.
_PVALUE_BLOCKS = {
    "dm_pvalue_by_variable": "random_walk",
    "dm_pvalue_vs_drift_by_variable": "drift",
    "dm_pvalue_vs_ar1_by_variable": "ar1",
}


def _add_fdr_qvalues(horizons: list[dict]) -> dict:
    """Attach BH q-values across the whole (variable, horizon) grid.

    Writes ``<key>_fdr_q`` next to each p-value block and returns a summary of
    how many tests survive at conventional FDR levels.
    """
    summary = {}
    for key, bench in _PVALUE_BLOCKS.items():
        if key not in horizons[0]:
            continue
        flat, index = [], []
        for h, row in enumerate(horizons):
            for var in COLUMNS:
                flat.append(row[key][var])
                index.append((h, var))
        q = benjamini_hochberg(flat)
        for (h, var), qv in zip(index, q):
            horizons[h].setdefault(key + "_fdr_q", {})[var] = (
                None if not np.isfinite(qv) else float(qv))
        finite = q[np.isfinite(q)]
        summary[bench] = {
            "tests": int(finite.size),
            "min_q": None if finite.size == 0 else float(finite.min()),
            "n_q_below_0.05": int((finite < 0.05).sum()),
            "n_q_below_0.10": int((finite < 0.10).sum()),
        }
    return summary

# Quarters the production model dummies out (scripts/run_replication.py,
# tests/conftest.py and the hosted adapter all use exactly this window).
COVID_START, COVID_END = "2020Q1", "2021Q2"


def covid_dummies(index):
    """One 0/1 column per Covid quarter, aligned with ``index``."""
    import pandas as pd

    quarters = pd.period_range(COVID_START, COVID_END, freq="Q")
    return np.column_stack([(index == q).astype(float) for q in quarters])


def _production_spec_block(df, lags: int, horizons: int, first_origin: int):
    """Rolling evaluation of the specification that is actually published.

    The headline block of this artifact evaluates a BVAR **without** the six
    Covid dummies, while every published object -- the hero fan chart, the
    replication summary, the hosted adapter -- estimates the model **with**
    them. Skill measured on the first is not skill of the second: leaving
    2020Q2 in the likelihood as an ordinary observation distorts the
    coefficients used at every origin from 2020 onward.

    The dummy columns are real-time safe by construction. A dummy for a
    quarter later than the origin is identically zero over the training
    sample, so it contributes nothing to the fit and leaves the point
    forecast numerically unchanged; only quarters the forecaster has already
    observed can matter. The remaining judgement -- that 2020Q1-2021Q2 is the
    pandemic window -- is the same judgement the production model makes.
    """
    import pandas as pd

    y = df.to_numpy()
    report = rolling_origin_evaluation(
        y, lags=lags, horizons=horizons, first_origin=first_origin,
        dummies=covid_dummies(df.index),
    )
    covid_lo = pd.Period(COVID_START, "Q")
    covid_hi = pd.Period(COVID_END, "Q")
    out = {
        "description": (
            "expanding-window pseudo-out-of-sample evaluation of the PUBLISHED "
            "specification: identical to the headline block except that the "
            f"six Covid dummies ({COVID_START}-{COVID_END}) enter as exogenous "
            "regressors, as they do in every published forecast"
        ),
        "covid_dummies": f"{COVID_START}-{COVID_END}",
        "origins": report["origins"],
        "horizons": [],
    }
    for h_idx, row in enumerate(report["horizons"], start=1):
        targets = [df.index[o + h_idx] for o in report["origin_index"]]
        keep = np.array([not (covid_lo <= t <= covid_hi) for t in targets])
        entry = {"horizon": row["horizon"], "n_origins": row["n_origins"]}
        for key in ("bvar_rmse", "random_walk_rmse", "drift_rmse", "ar1_rmse",
                    "relative_rmse", "relative_rmse_vs_drift",
                    "relative_rmse_vs_ar1", "dm_pvalue", "dm_pvalue_vs_drift",
                    "dm_pvalue_vs_ar1"):
            entry[f"{key}_by_variable"] = dict(zip(COLUMNS, row[key]))
        m = np.asarray(row["bvar_errors_by_origin"])[keep]
        ex = {}
        for label, bench in (("", "random_walk_errors_by_origin"),
                             ("_vs_drift", "drift_errors_by_origin"),
                             ("_vs_ar1", "ar1_errors_by_origin")):
            b = np.asarray(row[bench])[keep]
            br = np.sqrt(np.mean(b ** 2, axis=0))
            ratio = np.divide(np.sqrt(np.mean(m ** 2, axis=0)), br,
                              out=np.full(br.shape, np.nan), where=br > 0)
            ex[f"relative_rmse{label}_ex_covid_by_variable"] = dict(
                zip(COLUMNS, [None if not np.isfinite(v) else float(v)
                              for v in ratio]))
        entry["n_origins_ex_covid"] = int(keep.sum())
        entry.update(ex)
        entry["bvar_rmse_ex_covid_by_variable"] = dict(
            zip(COLUMNS, np.sqrt(np.mean(m ** 2, axis=0)).tolist()))
        out["horizons"].append(entry)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lags", type=int, default=4)
    parser.add_argument("--horizons", type=int, default=8)
    parser.add_argument("--first-origin", type=int, default=80)
    parser.add_argument("--output", default="results/rolling_evaluation.json")
    parser.add_argument("--start", default=None,
                        help="first quarter to use (e.g. 1992Q1); default: "
                             "full packaged dataset")
    parser.add_argument("--end", default=None,
                        help="last quarter to use; default: the data edge")
    args = parser.parse_args()

    import pandas as pd

    df = load_data()
    if args.start is not None:
        df = df.loc[df.index >= pd.Period(args.start, "Q")]
    if args.end is not None:
        df = df.loc[df.index <= pd.Period(args.end, "Q")]
    report = rolling_origin_evaluation(
        df.to_numpy(),
        lags=args.lags,
        horizons=args.horizons,
        first_origin=args.first_origin,
    )
    report["variables"] = COLUMNS
    report["sample"] = {"start": str(df.index[0]), "end": str(df.index[-1])}
    for row in report["horizons"]:
        row["relative_rmse_by_variable"] = dict(
            zip(COLUMNS, row.pop("relative_rmse"))
        )
        row["bvar_rmse_by_variable"] = dict(zip(COLUMNS, row.pop("bvar_rmse")))
        row["random_walk_rmse_by_variable"] = dict(
            zip(COLUMNS, row.pop("random_walk_rmse"))
        )
        for key in ("drift_rmse", "ar1_rmse", "relative_rmse_vs_drift",
                    "relative_rmse_vs_ar1", "dm_stat_vs_drift",
                    "dm_pvalue_vs_drift", "dm_stat_vs_ar1", "dm_pvalue_vs_ar1"):
            row[f"{key}_by_variable"] = dict(zip(COLUMNS, row.pop(key)))
        row["worst_origin_mse_share"] = {
            bench: dict(zip(COLUMNS, vals))
            for bench, vals in row["worst_origin_mse_share"].items()
        }
        row["dm_stat_by_variable"] = dict(zip(COLUMNS, row.pop("dm_stat")))
        row["dm_pvalue_by_variable"] = dict(zip(COLUMNS, row.pop("dm_pvalue")))
        # Per-origin errors keyed by variable, so downstream consumers do not
        # have to know the column order.
        for key in ("bvar_errors_by_origin", "random_walk_errors_by_origin",
                    "drift_errors_by_origin", "ar1_errors_by_origin",
                    "actuals_by_origin", "bvar_point_by_origin"):
            rows_by_origin = row.pop(key)
            row[key] = {
                col: [r[j] for r in rows_by_origin] for j, col in enumerate(COLUMNS)
            }
    # Episode split. Under squared loss a single -22 log-point observation
    # (2020Q2) can dominate a 49-origin average, so report the ratios with the
    # Covid target quarters excluded alongside the headline. This is a
    # decomposition, not a preferred number: both are published.
    covid_lo, covid_hi = pd.Period("2020Q1", "Q"), pd.Period("2021Q2", "Q")
    report["episode_split"] = {
        "excluded_range": "2020Q1-2021Q2",
        "note": "Ratios recomputed with target quarters in the excluded range "
                "dropped. Reported alongside, never instead of, the full sample.",
    }
    for h_idx, row in enumerate(report["horizons"], start=1):
        targets = [df.index[o + h_idx] for o in report["origin_index"]]
        keep = np.array([not (covid_lo <= t <= covid_hi) for t in targets])
        row["n_origins_ex_covid"] = int(keep.sum())
        for label, bench_key in (("", "random_walk_errors_by_origin"),
                                 ("_vs_drift", "drift_errors_by_origin"),
                                 ("_vs_ar1", "ar1_errors_by_origin")):
            out = {}
            for col in COLUMNS:
                m = np.asarray(row["bvar_errors_by_origin"][col])[keep]
                b = np.asarray(row[bench_key][col])[keep]
                br = np.sqrt(np.mean(b ** 2))
                out[col] = float(np.sqrt(np.mean(m ** 2)) / br) if br > 0 else None
            row[f"relative_rmse{label}_ex_covid_by_variable"] = out

    # The headline block above evaluates a BVAR with no Covid dummies, which
    # is NOT the specification any published forecast uses. Record the
    # published specification alongside it rather than silently swapping the
    # headline, so both are visible and the difference is auditable.
    report["limitations"].append(
        "The headline block estimates WITHOUT the six Covid dummies that "
        "every published forecast uses; see 'production_spec' for the "
        "like-for-like evaluation of the published specification."
    )
    report["production_spec"] = _production_spec_block(
        df, args.lags, args.horizons, args.first_origin
    )

    # Multiplicity control over the whole test grid, for both blocks.
    report["multiplicity"] = {
        "method": "Benjamini-Hochberg FDR over all variables x horizons, "
                  "computed separately for each benchmark",
        "note": "A p-value block without its q-values invites reading the "
                "smallest of 64 tests as a discovery.",
        "headline": _add_fdr_qvalues(report["horizons"]),
        "production_spec": _add_fdr_qvalues(report["production_spec"]["horizons"]),
    }
    report["limitations"].append(
        "No accuracy difference against the drifting random walk survives "
        "Benjamini-Hochberg control over the 8x8 test grid; see 'multiplicity'."
    )

    # Staleness guard: hash of the evaluation-relevant sources (plus this
    # script). tests/test_committed_artifacts.py fails if the code changes
    # without this artifact being regenerated.
    report["code_version"] = evaluation_code_version(extra_paths=[__file__])

    report["finite"] = bool(all(
        np.isfinite(list(row["relative_rmse_by_variable"].values())).all()
        for row in report["horizons"]
    ))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

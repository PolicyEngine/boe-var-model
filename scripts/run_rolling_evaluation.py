"""Generate leakage-safe rolling forecast evidence for the packaged dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from boe_var.data import COLUMNS, load_data
from boe_var.evaluation import rolling_origin_evaluation


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

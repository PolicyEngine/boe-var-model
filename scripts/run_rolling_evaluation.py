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
        row["dm_stat_by_variable"] = dict(zip(COLUMNS, row.pop("dm_stat")))
        row["dm_pvalue_by_variable"] = dict(zip(COLUMNS, row.pop("dm_pvalue")))
        # Per-origin errors keyed by variable, so downstream consumers do not
        # have to know the column order.
        for key in ("bvar_errors_by_origin", "random_walk_errors_by_origin"):
            rows_by_origin = row.pop(key)
            row[key] = {
                col: [r[j] for r in rows_by_origin] for j, col in enumerate(COLUMNS)
            }
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

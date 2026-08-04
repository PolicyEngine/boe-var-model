"""Interval-coverage evaluation for the BVAR's predictive bands.

The rolling-origin evaluation (``results/rolling_evaluation.json``) scores
point forecasts; this script scores the *bands*. At each expanding-window
origin it samples the BVAR posterior, simulates stochastic forecast paths
(parameter and shock uncertainty, the same recipe as the published fan
charts), forms 68% and 90% predictive intervals per variable and horizon,
and records how often the realised value fell inside. A calibrated model
covers ~68% and ~90%; materially more means the bands are too wide,
materially less too narrow.

Same leakage rules as the rolling evaluation: estimation uses data up to
and including the origin, forecasts start one quarter later, and estimation
uses final revised data -- pseudo- rather than real-time out-of-sample.
Structural identification is not used, so the check cannot be tuned to
match the paper's structural results.

Writes ``results/coverage_evaluation.json`` by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from boe_var import forecast as fc
from boe_var.bvar import BVAR
from boe_var.data import COLUMNS, load_data
from boe_var.evaluation import evaluation_code_version

LAGS = 4
HORIZONS = 8
FIRST_ORIGIN = 80          # matches rolling_evaluation.json
N_DRAWS = 100              # posterior draws per origin
N_PATHS = 5                # stochastic paths per draw
SEED = 20260729


def run_coverage(
    y: np.ndarray,
    *,
    lags: int = LAGS,
    horizons: int = HORIZONS,
    first_origin: int = FIRST_ORIGIN,
    n_draws: int = N_DRAWS,
    n_paths: int = N_PATHS,
    seed: int = SEED,
) -> tuple[dict, np.ndarray]:
    """Empirical band coverage. Returns (hits-per-level, origin count per h)."""
    T, k = y.shape
    last_origin = T - horizons - 1
    if first_origin <= lags + 1 or first_origin > last_origin:
        raise ValueError("evaluation window leaves no valid rolling origins")
    origins = list(range(first_origin, last_origin + 1))
    rng = np.random.default_rng(seed)

    # hits[level][h][var] counts outturns inside the level band
    hits = {lv: np.zeros((horizons, k)) for lv in (68, 90)}
    n = np.zeros(horizons)

    for origin in origins:
        train = y[: origin + 1]
        model = BVAR(train, lags=lags)
        draws = model.sample_posterior(n_draws, seed=seed + origin)
        paths = np.empty((n_draws * n_paths, horizons, k))
        i = 0
        for draw in draws:
            for _ in range(n_paths):
                paths[i] = fc.sample_forecast(
                    draw, train, horizons=horizons, rng=rng
                )
                i += 1
        actual = y[origin + 1: origin + horizons + 1]
        for lv, (a, b) in {68: (16.0, 84.0), 90: (5.0, 95.0)}.items():
            lo = np.percentile(paths, a, axis=0)
            hi = np.percentile(paths, b, axis=0)
            hits[lv] += (lo <= actual) & (actual <= hi)
        n += 1
    return hits, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lags", type=int, default=LAGS)
    parser.add_argument("--horizons", type=int, default=HORIZONS)
    parser.add_argument("--first-origin", type=int, default=FIRST_ORIGIN)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--paths", type=int, default=N_PATHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default="results/coverage_evaluation.json")
    args = parser.parse_args()

    df = load_data()
    y = df.to_numpy(float)
    T, k = y.shape
    n_origins = T - args.horizons - args.first_origin

    hits, n = run_coverage(
        y,
        lags=args.lags,
        horizons=args.horizons,
        first_origin=args.first_origin,
        n_draws=args.draws,
        n_paths=args.paths,
        seed=args.seed,
    )

    report = {
        "method": (
            "expanding-window pseudo-out-of-sample interval coverage; "
            "reduced-form BVAR predictive bands with parameter and shock "
            "uncertainty (posterior draws x stochastic paths), percentile "
            "intervals on the LEVEL of each series"
        ),
        "lags": args.lags,
        "horizons": args.horizons,
        "origins": n_origins,
        "first_origin": args.first_origin,
        "draws_per_origin": args.draws,
        "paths_per_draw": args.paths,
        "seed": args.seed,
        "sample": {"start": str(df.index[0]), "end": str(df.index[-1])},
        "limitations": [
            "Estimation uses final revised data (pseudo-, not real-time).",
            "Coverage is on series levels in transformed model units.",
            "Consecutive-origin outcomes overlap, so the effective number of "
            "independent observations is well below the origin count.",
            "No Covid dummies in the evaluation model, matching "
            "rolling_evaluation.json.",
        ],
        "coverage": {
            str(lv): {
                f"h{h + 1}": {
                    var: round(float(hits[lv][h, j] / n[h]), 4)
                    for j, var in enumerate(COLUMNS)
                }
                for h in range(args.horizons)
            }
            for lv in (68, 90)
        },
        # Staleness guard: hash of the evaluation-relevant sources (plus this
        # script). tests/test_committed_artifacts.py fails if the code changes
        # without this artifact being regenerated.
        "code_version": evaluation_code_version(extra_paths=[__file__]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n")
    lv68 = np.mean([hits[68][h].sum() / (n[h] * k) for h in range(args.horizons)])
    lv90 = np.mean([hits[90][h].sum() / (n[h] * k) for h in range(args.horizons)])
    print(
        f"wrote {output} -- mean coverage across variables/horizons: "
        f"68% band {lv68:.1%}, 90% band {lv90:.1%} ({n_origins} origins)"
    )


if __name__ == "__main__":
    main()

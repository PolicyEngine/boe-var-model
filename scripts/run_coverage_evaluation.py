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


def band_scale_factors(coverage: dict, levels=(68, 90)) -> dict:
    """Gaussian half-width correction implied by measured coverage.

    ``k(h, level) = z(nominal) / z(empirical)``, where ``z(p)`` is the
    standard-normal quantile at ``(1 + p) / 2``. Multiplying a band's
    half-width by ``k`` around its median rescales it to the width that would
    have delivered the nominal coverage under a Gaussian predictive density.
    ``k < 1`` narrows an over-covering band, so this is a calibration in both
    directions, not a one-way widening.

    ``coverage`` is the nested ``{level: {hN: {variable: fraction}}}`` mapping
    written into the artifact. Empirical coverage of 0 or 1 gives an infinite
    or zero factor and is returned as ``None`` rather than a fabricated
    number.

    The derivation is published here, next to the measurement it depends on,
    so a consumer applying these factors is not re-deriving them from a
    number whose provenance it cannot check.
    """
    from math import erf, sqrt

    def _z(p: float) -> float:
        # Inverse standard-normal CDF by bisection on erf; ample precision
        # here and avoids a scipy import in the artifact path.
        lo, hi = -12.0, 12.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    out = {}
    for lv in levels:
        z_nom = _z((1.0 + lv / 100.0) / 2.0)
        out[str(lv)] = {}
        for h_key, per_var in coverage[str(lv)].items():
            out[str(lv)][h_key] = {}
            for var, emp in per_var.items():
                if not 0.0 < emp < 1.0:
                    out[str(lv)][h_key][var] = None
                    continue
                out[str(lv)][h_key][var] = round(z_nom / _z((1.0 + emp) / 2.0), 4)
    return out


def run_coverage(
    y: np.ndarray,
    *,
    lags: int = LAGS,
    horizons: int = HORIZONS,
    first_origin: int = FIRST_ORIGIN,
    n_draws: int = N_DRAWS,
    n_paths: int = N_PATHS,
    seed: int = SEED,
    dummies: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    """Empirical band coverage. Returns (hits-per-level, origin count per h).

    ``dummies`` optionally supplies exogenous 0/1 columns aligned with ``y``
    (the six Covid quarters in the published specification). They are
    truncated to the training sample at each origin and set to zero over the
    forecast horizon, exactly as in the published forecast.
    """
    T, k = y.shape
    last_origin = T - horizons - 1
    if first_origin <= lags + 1 or first_origin > last_origin:
        raise ValueError("evaluation window leaves no valid rolling origins")
    if dummies is not None:
        dummies = np.asarray(dummies, dtype=float)
        if dummies.ndim == 1:
            dummies = dummies[:, None]
        if dummies.shape[0] != T:
            raise ValueError("dummies must have the same rows as y")
    origins = list(range(first_origin, last_origin + 1))
    rng = np.random.default_rng(seed)

    # hits[level][h][var] counts outturns inside the level band
    hits = {lv: np.zeros((horizons, k)) for lv in (68, 90)}
    n = np.zeros(horizons)

    for origin in origins:
        train = y[: origin + 1]
        train_dummies = None if dummies is None else dummies[: origin + 1]
        model = BVAR(train, lags=lags, dummies=train_dummies)
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

    import pandas as pd

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

    # The block above matches rolling_evaluation.json's headline: no Covid
    # dummies. Every PUBLISHED fan chart estimates with them, and dummying out
    # 2020 shrinks Sigma, so the published bands are narrower than the ones
    # measured here. A calibration factor derived from the block above and
    # applied to a published fan is therefore derived from a different model.
    # Measure the published specification too.
    covid_q = pd.period_range("2020Q1", "2021Q2", freq="Q")
    covid = np.column_stack([(df.index == q).astype(float) for q in covid_q])
    hits_prod, n_prod = run_coverage(
        y,
        lags=args.lags,
        horizons=args.horizons,
        first_origin=args.first_origin,
        n_draws=args.draws,
        n_paths=args.paths,
        seed=args.seed,
        dummies=covid,
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
            "No Covid dummies in the headline block, matching "
            "rolling_evaluation.json -- but every published forecast IS "
            "estimated with them; use 'production_spec' for the bands that "
            "are actually shown.",
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
    report["coverage_standard_error"] = {
        "note": (
            "Binomial standard error sqrt(p(1-p)/n) at the NOMINAL level with "
            f"n = {n_origins} origins, quoted so a measured coverage is not "
            "read as exact. Origins overlap, so the number of independent "
            "observations is well below n and this understates the true "
            "sampling error."
        ),
        "68": round(float(np.sqrt(0.68 * 0.32 / n_origins)), 4),
        "90": round(float(np.sqrt(0.90 * 0.10 / n_origins)), 4),
    }
    report["production_spec"] = {
        "description": (
            "same evaluation with the six Covid dummies (2020Q1-2021Q2) that "
            "every published forecast estimates with; this is the model whose "
            "bands are actually shown"
        ),
        "covid_dummies": "2020Q1-2021Q2",
        "coverage": {
            str(lv): {
                f"h{h + 1}": {
                    var: round(float(hits_prod[lv][h, j] / n_prod[h]), 4)
                    for j, var in enumerate(COLUMNS)
                }
                for h in range(args.horizons)
            }
            for lv in (68, 90)
        },
    }
    # Publish the derived half-width correction next to the measurement, so a
    # consumer does not have to re-derive it from a number it cannot check.
    report["band_scale_factors"] = {
        "derivation": (
            "k(h, level) = z(nominal) / z(empirical), z(p) the standard-normal "
            "quantile at (1+p)/2; multiply each band half-width by k around "
            "the median. k < 1 narrows an over-covering band."
        ),
        "caveat": (
            "Measured on series LEVELS in transformed model units. Published "
            "fans show year-on-year growth, whose forecast error at h > 4 is a "
            "difference of two level errors, so these factors are an "
            "approximation there. Apply the production_spec factors to a "
            "published fan; the headline factors describe a model with no "
            "Covid dummies."
        ),
        "headline": band_scale_factors(report["coverage"]),
        "production_spec": band_scale_factors(report["production_spec"]["coverage"]),
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

"""Held-out band evaluation of the Lenza-Primiceri Covid volatility scaling.

Opt-in experiment for issue #12; the published specification is unchanged.
Scores three reduced-form specifications on the same expanding-window
origins, seeds, draws and paths as ``run_coverage_evaluation.py``:

- ``headline``: no Covid treatment (``run_coverage`` default);
- ``covid_dummies``: the six 2020Q1-2021Q2 dummies every published fan uses;
- ``lp_scaling``: no dummies; innovation scales for 2020Q2-Q4 and a decay
  (``boe_var.covid_volatility``), re-estimated at every origin from the
  training sample only, and applied to the forecast shocks.

For each: 68% / 90% coverage, 90% half-width and CRPS by variable and
horizon, on all origins and on the origins #19 / #20 score, plus the
Gaussian-k (#19) and conformal (#20) recalibrations of each, so the volatility
model can be set against the post-hoc fixes on identical origins. Also the
share of explosive posterior draws (max companion root > 1) on the published
estimation sample under each Covid treatment.

Writes ``results/covid_volatility_evaluation.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_coverage_evaluation import (  # noqa: E402
    CONFORMAL_MIN_TRAIN,
    FIRST_ORIGIN,
    HORIZONS,
    LAGS,
    N_DRAWS,
    N_PATHS,
    SEED,
    TAUS,
    band_stretch,
    conformal_evaluation,
    holdout_calibration,
    quantile_crps,
    recalibrated_quantiles,
    run_coverage,
)

from boe_var import covid_volatility as cv  # noqa: E402
from boe_var.bvar import BVAR  # noqa: E402
from boe_var.data import COLUMNS, load_data  # noqa: E402
from boe_var.evaluation import evaluation_code_version  # noqa: E402

ROOTS_DRAWS = 10_000
ROOTS_SEED = 20260812          # results/summary.md seed
EST_START, EST_END = "1992Q1", "2025Q1"
CALM_FROM = "2022Q1"           # origins after the Covid outturns


def run_coverage_lp(y, index, *, lags=LAGS, horizons=HORIZONS,
                    first_origin=FIRST_ORIGIN, n_draws=N_DRAWS,
                    n_paths=N_PATHS, seed=SEED, estimate=True, record=None,
                    bvar_kwargs=None):
    """``run_coverage`` with the LP scaling re-estimated at every origin.

    Same RNG stream, origins and recording as ``run_coverage``. Each origin
    estimates (s0, s1, s2, rho) on ``y[:origin + 1]`` only and scales the
    forecast innovations by ``future_scale``. ``estimate=False`` holds every
    scale at 1, which must reproduce ``run_coverage`` bit for bit.
    ``record["scales"]`` gets the per-origin parameter estimates.
    """
    T, k = y.shape
    last_origin = T - horizons - 1
    origins = list(range(first_origin, last_origin + 1))
    rng = np.random.default_rng(seed)
    hits = {lv: np.zeros((horizons, k)) for lv in (68, 90)}
    n = np.zeros(horizons)
    stretch = {lv: [] for lv in (68, 90)}
    quant, acts, pit, scales = [], [], [], []
    kw = dict(bvar_kwargs or {})
    for origin in origins:
        train = y[: origin + 1]
        idx = index[: origin + 1]
        if estimate:
            p = cv.estimate_covid_scale(train, idx, lags=lags, bvar_kwargs=kw)
        else:
            p = {"s0": 1.0, "s1": 1.0, "s2": 1.0, "rho": cv.RHO_MODE,
                 "estimated": []}
        if p["estimated"]:
            model = cv.ScaledBVAR(train, lags=lags,
                                  scale=cv.scale_path(idx, p), **kw)
            fs = cv.future_scale(idx, p, horizons)
        else:
            model = BVAR(train, lags=lags, **kw)
            fs = None
        scales.append({"origin": str(index[origin]),
                       **{key: (round(float(p[key]), 4)
                                if key in p["estimated"] else None)
                          for key in ("s0", "s1", "s2", "rho")},
                       "future_scale_h1": None if fs is None
                       else round(float(fs[0]), 4)})
        draws = model.sample_posterior(n_draws, seed=seed + origin)
        paths = np.empty((n_draws * n_paths, horizons, k))
        i = 0
        for draw in draws:
            for _ in range(n_paths):
                paths[i] = cv.sample_forecast_scaled(
                    draw, train, horizons=horizons, rng=rng, shock_scale=fs)
                i += 1
        actual = y[origin + 1: origin + horizons + 1]
        med = np.percentile(paths, 50.0, axis=0)
        if record is not None:
            quant.append(np.moveaxis(
                np.percentile(paths, 100.0 * np.asarray(TAUS), axis=0), 0, -1))
            acts.append(actual)
            pit.append((paths <= actual[None]).mean(axis=0))
        for lv, (a, b) in {68: (16.0, 84.0), 90: (5.0, 95.0)}.items():
            lo = np.percentile(paths, a, axis=0)
            hi = np.percentile(paths, b, axis=0)
            hits[lv] += (lo <= actual) & (actual <= hi)
            if record is not None:
                stretch[lv].append(band_stretch(med, lo, hi, actual))
        n += 1
    if record is not None:
        record["stretch"] = {lv: np.asarray(v) for lv, v in stretch.items()}
        record["quantiles"] = np.asarray(quant)
        record["actual"] = np.asarray(acts)
        record["pit"] = np.asarray(pit)
        record["scales"] = scales
    return hits, n


def _grid(a, nd=4):
    """(origins, H, k) -> {var: {hN: mean}} plus by-horizon / by-var means."""
    H, k = a.shape[1], a.shape[2]
    return {
        "mean": round(float(np.nanmean(a)), nd),
        "by_horizon": {f"h{h + 1}": round(float(np.nanmean(a[:, h])), nd)
                       for h in range(H)},
        "by_variable": {v: round(float(np.nanmean(a[:, :, j])), nd)
                        for j, v in enumerate(COLUMNS[:k])},
        "by_variable_horizon": {
            v: {f"h{h + 1}": round(float(np.nanmean(a[:, h, j])), nd)
                for h in range(H)} for j, v in enumerate(COLUMNS[:k])},
    }


def summarise(rec, origin_dates):
    """Coverage, width and CRPS of one spec, raw and recalibrated."""
    Q, y = rec["quantiles"], rec["actual"]
    i05, i95 = TAUS.index(0.05), TAUS.index(0.95)
    i16, i84 = TAUS.index(0.16), TAUS.index(0.84)
    half90 = 0.5 * (Q[..., i95] - Q[..., i05])
    half68 = 0.5 * (Q[..., i84] - Q[..., i16])
    S68, S90 = rec["stretch"][68], rec["stretch"][90]
    calm = np.array([d >= pd.Period(CALM_FROM, "Q") for d in origin_dates])
    # origins #19/#20 score (first CONFORMAL_MIN_TRAIN usable ones dropped)
    n_o, H, k = S90.shape
    scored = np.zeros((n_o, H, k), bool)
    for h in range(H):
        scored[CONFORMAL_MIN_TRAIN + h:, h, :] = True
    Qr, fac = recalibrated_quantiles(Q, y)
    k90 = fac[0.9]
    crps_raw = quantile_crps(Q, y)
    okr = ~np.isnan(Qr).any(axis=-1)
    crps_conf = np.where(okr, quantile_crps(np.where(okr[..., None], Qr, 0.0),
                                            y), np.nan)
    sc = lambda a: np.where(scored, a, np.nan)  # noqa: E731
    calm_s = scored & calm[:, None, None]
    return {
        "coverage_all_origins": {"68": _grid((S68 <= 1).astype(float)),
                                 "90": _grid((S90 <= 1).astype(float))},
        "coverage_scored_origins": {
            "68": _grid(sc((S68 <= 1).astype(float))),
            "90": _grid(sc((S90 <= 1).astype(float)))},
        "half_width_90_all_origins": _grid(half90, 3),
        "half_width_68_all_origins": _grid(half68, 3),
        "half_width_90_calm_origins": {
            "raw": _grid(np.where(calm[:, None, None], half90, np.nan), 3),
            "conformal": _grid(np.where(calm_s, k90 * half90, np.nan), 3),
        },
        "crps_all_origins": _grid(crps_raw),
        "crps_scored_origins": {"raw": _grid(sc(crps_raw)),
                                "conformal": _grid(np.where(scored, crps_conf,
                                                            np.nan))},
        "pit_above_95_share_all_origins": round(float(
            (rec["pit"] > 0.95).mean()), 4),
        "pit_below_05_share_all_origins": round(float(
            (rec["pit"] < 0.05).mean()), 4),
        "gaussian_k_holdout": holdout_calibration(rec["stretch"]),
        "conformal_holdout": {lv: {key: v[key] for key in
                                   ("conformal", "gaussian_k", "raw",
                                    "half_width_factor_conformal",
                                    "half_width_factor_gaussian")}
                              for lv, v in conformal_evaluation(rec).items()
                              if lv in ("68", "90")},
    }


def explosive_share(seed=ROOTS_SEED, n_draws=ROOTS_DRAWS):
    """Share of reduced-form posterior draws with max |root| > 1 on the
    published estimation sample (theta = 1, lam 0.2, mu 1), per treatment."""
    import run_replication as rr

    df = load_data()
    df = df.loc[(df.index >= pd.Period(EST_START, "Q"))
                & (df.index <= pd.Period(EST_END, "Q"))]
    y = df.to_numpy(float)
    hyper = {"lam": 0.2, "mu": 1.0, "theta": 1.0}
    p = cv.estimate_covid_scale(y, df.index, bvar_kwargs=hyper)
    models = {
        "headline_no_covid_treatment": BVAR(y, lags=LAGS, **hyper),
        "covid_dummies": BVAR(y, lags=LAGS, dummies=rr.covid_dummies(df.index),
                              **hyper),
        "lp_scaling": cv.ScaledBVAR(y, lags=LAGS,
                                    scale=cv.scale_path(df.index, p), **hyper),
    }
    out = {"sample": f"{EST_START}-{EST_END}", "draws": n_draws, "seed": seed,
           "hyperparameters": hyper,
           "lp_scale_estimates": {key: round(float(p[key]), 4)
                                  for key in ("s0", "s1", "s2", "rho")},
           "lp_scale_path_2020Q2_to_end": [
               round(float(v), 3) for v in
               cv.scale_path(df.index, p)[df.index >= pd.Period(cv.COVID_START,
                                                                "Q")]],
           "log_marginal_likelihood": {}}
    for name, m in models.items():
        out["log_marginal_likelihood"][name] = round(
            float(m.log_marginal_likelihood()), 2)
        R = np.array([np.max(np.abs(np.linalg.eigvals(d.companion())))
                      for d in m.sample_posterior(n_draws, seed=seed)])
        out[name] = {"share_max_root_gt_1": round(float((R > 1).mean()), 4),
                     "share_gt_1.005": round(float((R > 1.005).mean()), 4),
                     "median_max_root": round(float(np.median(R)), 5),
                     "p5": round(float(np.percentile(R, 5)), 5),
                     "p95": round(float(np.percentile(R, 95)), 5)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--paths", type=int, default=N_PATHS)
    ap.add_argument("--output",
                    default="results/covid_volatility_evaluation.json")
    args = ap.parse_args()

    df = load_data()
    y = df.to_numpy(float)
    T, k = y.shape
    origins = list(range(FIRST_ORIGIN, T - HORIZONS))
    dates = [df.index[o] for o in origins]
    covid_q = pd.period_range("2020Q1", "2021Q2", freq="Q")
    covid = np.column_stack([(df.index == q).astype(float) for q in covid_q])

    recs = {"headline": {}, "covid_dummies": {}, "lp_scaling": {}}
    common = dict(lags=LAGS, horizons=HORIZONS, first_origin=FIRST_ORIGIN,
                  n_draws=args.draws, n_paths=args.paths, seed=SEED)
    run_coverage(y, record=recs["headline"], **common)
    run_coverage(y, dummies=covid, record=recs["covid_dummies"], **common)
    run_coverage_lp(y, df.index, record=recs["lp_scaling"], **common)

    report = {
        "method": (
            "expanding-window pseudo-out-of-sample band evaluation, same "
            "origins/seed/draws/paths as coverage_evaluation.json; reduced "
            "form, levels in model units. lp_scaling re-estimates the "
            "Lenza-Primiceri (2022) Covid innovation scales (2020Q2, Q3, Q4 "
            "free, geometric decay after; Pareto(1,1) / Beta(mode 0.8, sd "
            "0.2) hyperpriors, posterior mode) on each origin's training "
            "sample and scales the forecast shocks by 1 + (s_T - 1) rho^h"
        ),
        "origins": len(origins),
        "evaluated_origins": {"first": str(dates[0]), "last": str(dates[-1])},
        "calm_origins_from": CALM_FROM,
        "scored_origins_note": (
            "'scored' = the origins #19 (Gaussian k) and #20 (conformal) "
            f"score: the first {CONFORMAL_MIN_TRAIN} + h - 1 origins per "
            "horizon are their training burn-in"),
        "draws_per_origin": args.draws,
        "paths_per_draw": args.paths,
        "seed": SEED,
        "limitations": [
            "Scale parameters at their posterior mode, not integrated out "
            "(LP sample them by MCMC); lam/mu/theta fixed at the coverage "
            "evaluation's defaults (theta off), as in coverage_evaluation.json.",
            "Final revised data (pseudo-, not real-time).",
            "Consecutive origins share outturn quarters; ~30-49 origins, so "
            "90% coverage has a sampling s.e. of about 0.05.",
            "At origins 2020Q2 and 2020Q3 the forecast decays from the last "
            "in-sample scale with rho at its prior mode (0.8), since the "
            "later scales and rho are not yet identified.",
        ],
        "specs": {name: summarise(rec, dates) for name, rec in recs.items()},
        "lp_scale_by_origin": recs["lp_scaling"]["scales"],
        "explosive_draws": explosive_share(),
        "code_version": evaluation_code_version(
            extra_paths=[cv.__file__, __file__,
                         Path(__file__).with_name("run_coverage_evaluation.py")]),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    for name, s in report["specs"].items():
        c = s["coverage_all_origins"]
        print(f"{name}: 68% {c['68']['mean']:.3f}  90% {c['90']['mean']:.3f}"
              f"  crps {s['crps_all_origins']['mean']:.3f}")
    print(json.dumps({key: v for key, v in report["explosive_draws"].items()
                      if key not in ("lp_scale_path_2020Q2_to_end",)},
                     indent=1))


if __name__ == "__main__":
    main()

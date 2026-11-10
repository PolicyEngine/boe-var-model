"""Held-out band evaluation of the variable-specific SV BVAR (Cogley-Sargent).

Opt-in experiment for issue #12; the published specification is unchanged.
Scores ``boe_var.variable_sv`` (one random-walk log-volatility per
reduced-form shock, triangular A) on the same expanding-window origins,
seeds, draws and paths as ``run_coverage_evaluation.py``,
``run_covid_volatility_evaluation.py`` and ``run_common_sv_evaluation.py``:

- ``vsv_rw``: no Covid dummies;
- ``vsv_rw_dummies``: the six 2020Q1-2021Q2 dummies the published fan uses.

The comparison specs (published = ``covid_dummies``, ``lp_scaling``,
``sv_rw``, ``sv_rw_dummies``) are read from the committed
``results/common_sv_evaluation.json`` (same origins and seeds) rather than
re-run. For each spec: 68 / 90% coverage, half-widths and CRPS by variable
and horizon, the Gaussian-k (#19) and conformal (#20) recalibrations, and
ratios against the published spec. Also the full-sample posterior: per-
variable volatility at selected quarters, sigma2_i and explosive shares.

Writes ``results/variable_sv_evaluation.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_common_sv_evaluation import covid_dummy_matrix  # noqa: E402
from run_coverage_evaluation import (  # noqa: E402
    FIRST_ORIGIN,
    HORIZONS,
    LAGS,
    N_DRAWS,
    N_PATHS,
    SEED,
    TAUS,
    band_stretch,
)
from run_covid_volatility_evaluation import (  # noqa: E402
    CALM_FROM,
    EST_END,
    EST_START,
    summarise,
)

from boe_var import variable_sv as vs  # noqa: E402
from boe_var.data import COLUMNS, load_data  # noqa: E402
from boe_var.evaluation import evaluation_code_version  # noqa: E402

BURN, THIN = 1000, 10
ROOTS_SEED = 20260812
RESULTS = Path(__file__).resolve().parents[1] / "results"
COMPARE = {"published": "covid_dummies", "lp_scaling": "lp_scaling",
           "common_sv_rw": "sv_rw", "common_sv_rw_dummies": "sv_rw_dummies"}


def run_coverage_vsv(y, *, lags=LAGS, horizons=HORIZONS,
                     first_origin=FIRST_ORIGIN, n_draws=N_DRAWS,
                     n_paths=N_PATHS, seed=SEED, dummies=None,
                     fix_volatility=False, burn=BURN, thin=THIN, record=None,
                     bvar_kwargs=None):
    """``run_coverage`` with the variable-SV BVAR re-estimated per origin.

    Same origins, RNG stream and recording as ``run_coverage_sv``. Each
    origin runs the Gibbs sampler on ``y[:origin + 1]`` only (seed
    ``seed + origin``); ``record["volatility"]`` gets per-origin summaries.
    """
    T, k = y.shape
    last_origin = T - horizons - 1
    if dummies is not None:
        dummies = np.asarray(dummies, dtype=float)
        if dummies.ndim == 1:
            dummies = dummies[:, None]
    rng = np.random.default_rng(seed)
    hits = {lv: np.zeros((horizons, k)) for lv in (68, 90)}
    n = np.zeros(horizons)
    stretch = {lv: [] for lv in (68, 90)}
    quant, acts, pit, vol = [], [], [], []
    kw = dict(bvar_kwargs or {})
    for origin in range(first_origin, last_origin + 1):
        train = y[: origin + 1]
        train_dummies = None if dummies is None else dummies[: origin + 1]
        model = vs.VariableSVBVAR(train, lags=lags, dummies=train_dummies,
                                  **kw)
        draws = model.sample(n_draws, seed=seed + origin, burn=burn,
                             thin=thin, fix_volatility=fix_volatility)
        vol.append({
            "h_T_mean": [round(float(v), 4) for v in
                         np.mean([d.h_T for d in draws], axis=0)],
            "sigma2_mean": [round(float(v), 4) for v in
                            np.mean([d.sigma2 for d in draws], axis=0)],
        })
        paths = np.empty((n_draws * n_paths, horizons, k))
        i = 0
        for draw in draws:
            for _ in range(n_paths):
                paths[i] = vs.sample_forecast_vsv(draw, train,
                                                  horizons=horizons, rng=rng)
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
        record["volatility"] = vol
    return hits, n


def full_sample_vsv(n_draws=1000, seed=ROOTS_SEED, burn=BURN, thin=THIN):
    """Full-sample posterior, published hyperparameters, with/without dummies."""
    df = load_data()
    df = df.loc[(df.index >= pd.Period(EST_START, "Q"))
                & (df.index <= pd.Period(EST_END, "Q"))]
    y = df.to_numpy(float)
    k = y.shape[1]
    hyper = {"lam": 0.2, "mu": 1.0, "theta": 1.0}
    out = {"sample": f"{EST_START}-{EST_END}", "hyperparameters": hyper,
           "kept_draws": n_draws, "burn": burn, "thin": thin, "seed": seed}
    idx = list(df.index[LAGS:])
    show = ["2008Q4", "2019Q4", "2020Q2", "2021Q2", "2022Q1", "2022Q4",
            "2023Q4", "2025Q1"]
    pos = [idx.index(pd.Period(q, "Q")) for q in show]
    for name, use_d in {"vsv_rw_dummies": True, "vsv_rw": False}.items():
        D = covid_dummy_matrix(df.index) if use_d else None
        m = vs.VariableSVBVAR(y, lags=LAGS, dummies=D, **hyper)
        draws = m.sample(n_draws, seed=seed, burn=burn, thin=thin)
        R = np.array([np.max(np.abs(np.linalg.eigvals(d.companion())))
                      for d in draws])
        out[name] = {
            "variance_ratio_posterior_median": {
                v: {q: round(float(np.exp(m.h_bands[1, p, j])), 3)
                    for q, p in zip(show, pos)}
                for j, v in enumerate(COLUMNS[:k])},
            "sigma2_mean": {v: round(float(s), 4) for v, s in zip(
                COLUMNS[:k], np.mean([d.sigma2 for d in draws], axis=0))},
            "share_max_root_gt_1": round(float((R > 1).mean()), 4),
            "share_gt_1.005": round(float((R > 1.005).mean()), 4),
            "median_max_root": round(float(np.median(R)), 5),
        }
    return out


def _mean(x):
    return x["mean"] if isinstance(x, dict) else x


def _ratio(a, b):
    return {v: {h: round(a[v][h] / b[v][h], 3) for h in a[v]} for v in a}


def compare(specs):
    """Headline means and ratios against the published spec."""
    pub = specs["published"]
    out = {}
    for name, s in specs.items():
        g, c = s["gaussian_k_holdout"], s["conformal_holdout"]
        out[name] = {
            "coverage_68": s["coverage_all_origins"]["68"]["mean"],
            "coverage_90": s["coverage_all_origins"]["90"]["mean"],
            "coverage_90_by_variable":
                s["coverage_all_origins"]["90"]["by_variable"],
            "coverage_90_gaussian_k": _mean(
                g["90"]["rolling_origin_calibrated"]),
            "coverage_90_conformal": _mean(c["90"]["conformal"]),
            "half_width_90": s["half_width_90_all_origins"]["mean"],
            "width_ratio_vs_published": round(
                s["half_width_90_all_origins"]["mean"]
                / pub["half_width_90_all_origins"]["mean"], 3),
            "width_ratio_vs_published_by_variable": {
                v: round(x / pub["half_width_90_all_origins"]["by_variable"][v],
                         3) for v, x in
                s["half_width_90_all_origins"]["by_variable"].items()},
            "crps_ratio_vs_published": round(
                s["crps_all_origins"]["mean"]
                / pub["crps_all_origins"]["mean"], 3),
            "crps_ratio_vs_published_by_variable": {
                v: round(x / pub["crps_all_origins"]["by_variable"][v], 3)
                for v, x in s["crps_all_origins"]["by_variable"].items()},
            "crps_ratio_vs_published_by_variable_horizon": _ratio(
                s["crps_all_origins"]["by_variable_horizon"],
                pub["crps_all_origins"]["by_variable_horizon"]),
        }
    return out


def _run_spec(args):
    name, use_dummies, common, burn, thin = args
    df = load_data()
    y = df.to_numpy(float)
    T = y.shape[0]
    dates = [df.index[o] for o in range(FIRST_ORIGIN, T - HORIZONS)]
    rec = {}
    run_coverage_vsv(y, dummies=covid_dummy_matrix(df.index)
                     if use_dummies else None, burn=burn, thin=thin,
                     record=rec, **common)
    vol = [{"origin": str(d), **v} for d, v in zip(dates, rec["volatility"])]
    return name, summarise(rec, dates), vol


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--paths", type=int, default=N_PATHS)
    ap.add_argument("--burn", type=int, default=BURN)
    ap.add_argument("--thin", type=int, default=THIN)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--full-draws", type=int, default=1000)
    ap.add_argument("--output", default="results/variable_sv_evaluation.json")
    args = ap.parse_args()

    df = load_data()
    T, k = df.shape
    origins = list(range(FIRST_ORIGIN, T - HORIZONS))
    dates = [df.index[o] for o in origins]
    common = dict(lags=LAGS, horizons=HORIZONS, first_origin=FIRST_ORIGIN,
                  n_draws=args.draws, n_paths=args.paths, seed=SEED)
    jobs = [("vsv_rw", False, common, args.burn, args.thin),
            ("vsv_rw_dummies", True, common, args.burn, args.thin)]
    if args.workers > 1:
        ctx = multiprocessing.get_context("spawn")      # fork + BLAS can hang
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=ctx) as ex:
            res = list(ex.map(_run_spec, jobs))
    else:
        res = [_run_spec(j) for j in jobs]
    ref = json.loads((RESULTS / "common_sv_evaluation.json").read_text())
    specs = {label: ref["specs"][key] for label, key in COMPARE.items()}
    vol = {}
    for name, summ, v in res:
        specs[name] = summ
        vol[name] = v

    report = {
        "method": (
            "expanding-window pseudo-out-of-sample band evaluation, same "
            "origins/seed/draws/paths as common_sv_evaluation.json; reduced "
            "form, levels in model units. vsv_* re-run the Cogley-Sargent / "
            "Clark (2011) variable-specific stochastic volatility Gibbs "
            "sampler (joint GLS draw of Pi, triangular A and scales d, KSC "
            "mixture + FFBS for each random-walk log-volatility, IG sigma2_i) "
            "on each origin's training sample and simulate each "
            "log-volatility forward over the forecast horizon. Comparison "
            "specs are copied from common_sv_evaluation.json."),
        "priors": {"sigma2_i": f"IG{vs.SIGMA2_PRIOR}",
                   "Sigma_reference": "|Sigma|^-(k+1)/2 (BVAR's NIW)",
                   "h_i0": "pinned at 0 (pre-sample quarter)",
                   "Pi": "BVAR dummy rows at the reference covariance"},
        "gibbs": {"burn": args.burn, "thin": args.thin},
        "origins": len(origins),
        "evaluated_origins": {"first": str(dates[0]), "last": str(dates[-1])},
        "calm_origins_from": CALM_FROM,
        "draws_per_origin": args.draws,
        "paths_per_draw": args.paths,
        "seed": SEED,
        "limitations": [
            "lam/mu/theta fixed at the coverage evaluation's defaults, as in "
            "coverage_evaluation.json and common_sv_evaluation.json.",
            "Random walk log-volatility only (no AR(1), no fat tails); the "
            "Covid quarters inflate sigma2_i without dummies.",
            "A is triangular in data order (world_gdp first, uk_gdp last); "
            "the reduced-form predictive density is order-dependent under "
            "SV. No ordering sweep was run.",
            "100 kept draws per origin (thinned 1 in 10 after 1000 burn-in "
            "sweeps); no convergence diagnostics per origin.",
            "Final revised data (pseudo-, not real-time). 49 origins, so 90% "
            "coverage has a sampling s.e. of about 0.04-0.05.",
        ],
        "comparison": compare(specs),
        "specs": {name: specs[name] for name in ("vsv_rw", "vsv_rw_dummies")},
        "volatility_by_origin": vol,
        "full_sample_posterior": full_sample_vsv(
            n_draws=args.full_draws, burn=args.burn, thin=args.thin),
        "variables": list(COLUMNS[:k]),
        "code_version": evaluation_code_version(
            extra_paths=[vs.__file__, __file__,
                         RESULTS / "common_sv_evaluation.json",
                         Path(__file__).with_name("run_common_sv_evaluation.py"),
                         Path(__file__).with_name("run_coverage_evaluation.py"),
                         Path(__file__).with_name(
                             "run_covid_volatility_evaluation.py")]),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    for name, c in report["comparison"].items():
        print(f"{name}: 68% {c['coverage_68']:.3f}  90% {c['coverage_90']:.3f}"
              f"  +k {c['coverage_90_gaussian_k']:.3f}"
              f"  width x{c['width_ratio_vs_published']:.2f}"
              f"  crps x{c['crps_ratio_vs_published']:.3f}")


if __name__ == "__main__":
    main()

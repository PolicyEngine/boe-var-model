"""Held-out band evaluation of the common stochastic volatility BVAR (CCM 2016).

Opt-in experiment for issue #12; the published specification is unchanged.
Scores, on the same expanding-window origins, seeds, draws and paths as
``run_coverage_evaluation.py`` and ``run_covid_volatility_evaluation.py``:

- ``headline``: homoskedastic, no Covid treatment (``run_coverage`` default);
- ``covid_dummies``: homoskedastic with the six 2020Q1-2021Q2 dummies every
  published fan uses;
- ``lp_scaling``: Lenza-Primiceri Covid scaling (#21);
- ``sv_rw`` / ``sv_ar1``: common SV, random-walk / AR(1) log-volatility, no
  dummies (``boe_var.common_sv``), re-estimated by Gibbs at every origin on
  the training sample only, with the forecast volatility simulated forward;
- ``sv_rw_dummies`` / ``sv_ar1_dummies``: the same with the six dummies.

For each: 68% / 90% coverage, half-widths and CRPS by variable and horizon,
plus the Gaussian-k (#19) and conformal (#20) recalibrations of each spec
(``summarise`` from the LP script). Also the full-sample (1992Q1-2025Q1,
published hyperparameters) SV posterior: log-volatility path, sigma2, phi,
and the share of explosive draws.

Writes ``results/common_sv_evaluation.json``.
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
    FIRST_ORIGIN,
    HORIZONS,
    LAGS,
    N_DRAWS,
    N_PATHS,
    SEED,
    TAUS,
    band_stretch,
    run_coverage,
)
from run_covid_volatility_evaluation import (  # noqa: E402
    CALM_FROM,
    EST_END,
    EST_START,
    run_coverage_lp,
    summarise,
)

from boe_var import common_sv as sv  # noqa: E402
from boe_var.data import COLUMNS, load_data  # noqa: E402
from boe_var.evaluation import evaluation_code_version  # noqa: E402

BURN, THIN = 1000, 10
ROOTS_SEED = 20260812


def covid_dummy_matrix(index) -> np.ndarray:
    q = pd.period_range("2020Q1", "2021Q2", freq="Q")
    return np.column_stack([(index == p).astype(float) for p in q])


def run_coverage_sv(y, *, lags=LAGS, horizons=HORIZONS,
                    first_origin=FIRST_ORIGIN, n_draws=N_DRAWS,
                    n_paths=N_PATHS, seed=SEED, dummies=None, law="rw",
                    fix_volatility=False, burn=BURN, thin=THIN, record=None,
                    bvar_kwargs=None):
    """``run_coverage`` with the common-SV BVAR re-estimated at every origin.

    Same RNG stream, origins, dummy handling and recording as
    ``run_coverage``. Each origin runs the Gibbs sampler on
    ``y[:origin + 1]`` only (seed ``seed + origin``). ``fix_volatility=True``
    holds lambda_t = 1 and must reproduce ``run_coverage`` bit for bit.
    ``record["volatility"]`` gets per-origin posterior summaries.
    """
    T, k = y.shape
    last_origin = T - horizons - 1
    if dummies is not None:
        dummies = np.asarray(dummies, dtype=float)
        if dummies.ndim == 1:
            dummies = dummies[:, None]
    origins = list(range(first_origin, last_origin + 1))
    rng = np.random.default_rng(seed)
    hits = {lv: np.zeros((horizons, k)) for lv in (68, 90)}
    n = np.zeros(horizons)
    stretch = {lv: [] for lv in (68, 90)}
    quant, acts, pit, vol = [], [], [], []
    kw = dict(bvar_kwargs or {})
    for origin in origins:
        train = y[: origin + 1]
        train_dummies = None if dummies is None else dummies[: origin + 1]
        model = sv.CommonSVBVAR(train, lags=lags, dummies=train_dummies,
                                law=law, **kw)
        draws = model.sample(n_draws, seed=seed + origin, burn=burn,
                             thin=thin, fix_volatility=fix_volatility)
        vol.append({
            "h_T_mean": round(float(np.mean([d.h_T for d in draws])), 4),
            "sigma2_mean": round(float(np.mean([d.sigma2 for d in draws])), 4),
            "phi_mean": round(float(np.mean([d.phi for d in draws])), 4),
        })
        paths = np.empty((n_draws * n_paths, horizons, k))
        i = 0
        for draw in draws:
            for _ in range(n_paths):
                paths[i] = sv.sample_forecast_sv(draw, train,
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


def full_sample_sv(n_draws=1000, seed=ROOTS_SEED):
    """Full-sample SV posterior, published hyperparameters and dummies."""
    df = load_data()
    df = df.loc[(df.index >= pd.Period(EST_START, "Q"))
                & (df.index <= pd.Period(EST_END, "Q"))]
    y = df.to_numpy(float)
    hyper = {"lam": 0.2, "mu": 1.0, "theta": 1.0}
    out = {"sample": f"{EST_START}-{EST_END}", "hyperparameters": hyper,
           "kept_draws": n_draws, "burn": BURN, "thin": THIN, "seed": seed}
    idx = df.index[LAGS:]
    show = ["1995Q1", "2000Q1", "2008Q4", "2009Q2", "2016Q3", "2019Q4",
            "2020Q2", "2020Q3", "2021Q2", "2022Q1", "2022Q4", "2023Q4",
            "2024Q4", "2025Q1"]
    pos = [list(idx).index(pd.Period(q, "Q")) for q in show]
    for name, kw in {"sv_rw_dummies": {"law": "rw"},
                     "sv_ar1_dummies": {"law": "ar1"},
                     "sv_rw": {"law": "rw", "no_dummies": True}}.items():
        kw = dict(kw)
        D = None if kw.pop("no_dummies", False) else covid_dummy_matrix(
            df.index)
        m = sv.CommonSVBVAR(y, lags=LAGS, dummies=D, **kw, **hyper)
        draws = m.sample(n_draws, seed=seed)
        R = np.array([np.max(np.abs(np.linalg.eigvals(d.companion())))
                      for d in draws])
        out[name] = {
            "lambda_posterior_median_and_90pct": {
                q: [round(float(np.exp(m.h_bands[j, p])), 3)
                    for j in (1, 0, 2)] for q, p in zip(show, pos)},
            "sigma2_mean": round(float(np.mean([d.sigma2 for d in draws])), 4),
            "phi_mean": round(float(np.mean([d.phi for d in draws])), 4),
            "share_max_root_gt_1": round(float((R > 1).mean()), 4),
            "share_gt_1.005": round(float((R > 1.005).mean()), 4),
            "median_max_root": round(float(np.median(R)), 5),
        }
    return out


SPECS = {
    "sv_rw": {"law": "rw", "dummies": False},
    "sv_ar1": {"law": "ar1", "dummies": False},
    "sv_rw_dummies": {"law": "rw", "dummies": True},
    "sv_ar1_dummies": {"law": "ar1", "dummies": True},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--paths", type=int, default=N_PATHS)
    ap.add_argument("--burn", type=int, default=BURN)
    ap.add_argument("--thin", type=int, default=THIN)
    ap.add_argument("--output", default="results/common_sv_evaluation.json")
    args = ap.parse_args()

    df = load_data()
    y = df.to_numpy(float)
    T, k = y.shape
    origins = list(range(FIRST_ORIGIN, T - HORIZONS))
    dates = [df.index[o] for o in origins]
    covid = covid_dummy_matrix(df.index)

    common = dict(lags=LAGS, horizons=HORIZONS, first_origin=FIRST_ORIGIN,
                  n_draws=args.draws, n_paths=args.paths, seed=SEED)
    recs = {"headline": {}, "covid_dummies": {}, "lp_scaling": {}}
    run_coverage(y, record=recs["headline"], **common)
    run_coverage(y, dummies=covid, record=recs["covid_dummies"], **common)
    run_coverage_lp(y, df.index, record=recs["lp_scaling"], **common)
    vol = {}
    for name, spec in SPECS.items():
        recs[name] = {}
        run_coverage_sv(y, dummies=covid if spec["dummies"] else None,
                        law=spec["law"], burn=args.burn, thin=args.thin,
                        record=recs[name], **common)
        vol[name] = [{"origin": str(d), **v}
                     for d, v in zip(dates, recs[name]["volatility"])]

    report = {
        "method": (
            "expanding-window pseudo-out-of-sample band evaluation, same "
            "origins/seed/draws/paths as coverage_evaluation.json and "
            "covid_volatility_evaluation.json; reduced form, levels in model "
            "units. sv_* re-run the Carriero-Clark-Marcellino (2016) common "
            "stochastic volatility Gibbs sampler (KSC mixture + FFBS for the "
            "log-volatility, NIW for (Pi, Sigma | h), IG for sigma2, truncated "
            "normal for phi) on each origin's training sample and simulate "
            "the log-volatility forward over the forecast horizon"),
        "priors": {"sigma2": f"IG{sv.SIGMA2_PRIOR}",
                   "phi_ar1": f"N{sv.PHI_PRIOR} truncated to |phi|<1",
                   "h_0": "pinned at 0 (pre-sample quarter)"},
        "gibbs": {"burn": args.burn, "thin": args.thin},
        "origins": len(origins),
        "evaluated_origins": {"first": str(dates[0]), "last": str(dates[-1])},
        "calm_origins_from": CALM_FROM,
        "draws_per_origin": args.draws,
        "paths_per_draw": args.paths,
        "seed": SEED,
        "limitations": [
            "lam/mu/theta fixed at the coverage evaluation's defaults (theta "
            "off), as in coverage_evaluation.json; the published spec uses "
            "theta = 1.",
            "Gaussian log-volatility innovations: a random walk cannot jump "
            "to the 2020Q2 variance in one quarter, so without dummies the "
            "Covid spike inflates sigma2 and smears into the neighbouring "
            "quarters.",
            "The level split between lambda_t and Sigma rests on h_0 = 0 and "
            "the IW dummy prior (AR(1) scales from the training sample, "
            "Covid-inflated after 2020); the product lambda_t Sigma is what "
            "the forecasts use.",
            "100 kept draws per origin (thinned 1 in 10 after 1000 burn-in "
            "sweeps); no convergence diagnostics per origin.",
            "Final revised data (pseudo-, not real-time). ~30-49 origins, so "
            "90% coverage has a sampling s.e. of about 0.05.",
        ],
        "specs": {name: summarise(rec, dates) for name, rec in recs.items()},
        "volatility_by_origin": vol,
        "full_sample_posterior": full_sample_sv(),
        "variables": list(COLUMNS[:k]),
        "code_version": evaluation_code_version(
            extra_paths=[sv.__file__, __file__,
                         Path(__file__).with_name("run_coverage_evaluation.py"),
                         Path(__file__).with_name(
                             "run_covid_volatility_evaluation.py")]),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    for name, s in report["specs"].items():
        c = s["coverage_all_origins"]
        g = s["gaussian_k_holdout"]
        print(f"{name}: 68% {c['68']['mean']:.3f}  90% {c['90']['mean']:.3f}"
              f"  crps {s['crps_all_origins']['mean']:.3f}"
              f"  hw90 {s['half_width_90_all_origins']['mean']:.3f}"
              f"  k-holdout {json.dumps(g)[:120]}")


if __name__ == "__main__":
    main()

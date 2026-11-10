"""Explosive posterior draws vs the sum-of-coefficients tightness (issue #11).

Report only; the published hyperparameters (lam 0.2, mu 1, theta 1) are
unchanged. Brignone & Piffer (2025, p.10 and fn 4) say they follow Giannone,
Lenza & Primiceri (2015) with a random-walk Minnesota prior and the
sum-of-coefficients prior, and give no numerical values; the repo's
lam = 0.2, mu = 1, theta = 1 are GLP's hyperprior modes. GLP themselves
select the hyperparameters by marginal likelihood, so the ML-optimal values
on this sample are reported too.

For each (mu, theta) on the published sample (1992Q1-2025Q1, six Covid
dummies, lam = 0.2):

- share of reduced-form posterior draws with max |companion root| > 1 and
  > 1.005 (the first is partly a max-of-eight-near-unit-roots artefact of a
  prior centred on a random walk), median max root;
- the same share under the prior alone (NIW from the dummy observations);
- log marginal likelihood (in-sample fit; comparable across mu, theta);
- held-out 68 / 90 coverage and CRPS on the 49 coverage-evaluation origins
  (same seeds / draws / paths as coverage_evaluation.json, dummies on).

Writes ``results/explosive_roots_sweep.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import invwishart

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_common_sv_evaluation import covid_dummy_matrix, run_coverage_sv  # noqa: E402
from run_coverage_evaluation import (  # noqa: E402
    FIRST_ORIGIN,
    HORIZONS,
    LAGS,
    N_DRAWS,
    N_PATHS,
    SEED,
    quantile_crps,
)
from run_covid_volatility_evaluation import EST_END, EST_START  # noqa: E402

from boe_var import common_sv as sv  # noqa: E402
from boe_var.bvar import BVAR, PosteriorDraw, optimize_hyperparameters  # noqa: E402
from boe_var.data import COLUMNS, load_data  # noqa: E402
from boe_var.evaluation import evaluation_code_version  # noqa: E402

ROOTS_SEED = 20260812
MU_GRID = (1.0, 2.0, 5.0, 10.0, 1e6)       # 1e6 ~ prior switched off
THETA_GRID = (1.0, None)


def max_roots(draws) -> np.ndarray:
    return np.array([np.max(np.abs(np.linalg.eigvals(d.companion())))
                     for d in draws])


def prior_draws(model: BVAR, n: int, seed: int):
    """(Pi, Sigma) from the NIW implied by the dummy observations alone."""
    Td = model._Ys.shape[0] - model.Y.shape[0]
    Yd, Xd = model._Ys[-Td:], model._Xs[-Td:]
    XtX = Xd.T @ Xd
    C = cholesky(XtX, lower=True)
    B0 = solve_triangular(C.T, solve_triangular(C, Xd.T @ Yd, lower=True),
                          lower=False)
    E = Yd - Xd @ B0
    S0, d0 = E.T @ E, Td - model.m
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        Sig = np.atleast_2d(invwishart.rvs(df=d0, scale=S0, random_state=rng))
        Ls = cholesky(Sig, lower=True)
        V = solve_triangular(C.T, rng.standard_normal((model.m, model.k)),
                             lower=False)
        out.append(PosteriorDraw(Pi=(B0 + V @ Ls.T).T, Sigma=Sig,
                                 lags=model.lags, k=model.k))
    return out, int(d0)


def summary(R):
    return {"share_max_root_gt_1": round(float((R > 1).mean()), 4),
            "share_gt_1.005": round(float((R > 1.005).mean()), 4),
            "median_max_root": round(float(np.median(R)), 5),
            "p95": round(float(np.percentile(R, 95)), 5)}


def _holdout(yf, Df, mu):
    rec = {}
    run_coverage_sv(yf, lags=LAGS, horizons=HORIZONS,
                    first_origin=FIRST_ORIGIN, n_draws=N_DRAWS,
                    n_paths=N_PATHS, seed=SEED, dummies=Df,
                    fix_volatility=True, record=rec,
                    bvar_kwargs={"mu": mu, "theta": None})
    S = {lv: rec["stretch"][lv] for lv in (68, 90)}
    crps = quantile_crps(rec["quantiles"], rec["actual"])
    return {
        "coverage_68": round(float((S[68] <= 1).mean()), 4),
        "coverage_90": round(float((S[90] <= 1).mean()), 4),
        "crps_mean": round(float(crps.mean()), 4),
        "crps_by_variable": {v: round(float(crps[:, :, j].mean()), 4)
                             for j, v in enumerate(COLUMNS[: yf.shape[1]])},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=10_000)
    ap.add_argument("--output", default="results/explosive_roots_sweep.json")
    args = ap.parse_args()

    full = load_data()
    df = full.loc[(full.index >= pd.Period(EST_START, "Q"))
                  & (full.index <= pd.Period(EST_END, "Q"))]
    y = df.to_numpy(float)
    D = covid_dummy_matrix(df.index)
    yf = full.to_numpy(float)
    Df = covid_dummy_matrix(full.index)

    rows, held = [], {}
    for theta in THETA_GRID:
        for mu in MU_GRID:
            m = BVAR(y, lags=LAGS, dummies=D, lam=0.2, mu=mu, theta=theta)
            post = summary(max_roots(m.sample_posterior(args.draws,
                                                        seed=ROOTS_SEED)))
            pdraws, d0 = prior_draws(m, min(args.draws, 2000), ROOTS_SEED)
            if mu not in held:
                held[mu] = _holdout(yf, Df, mu)
            rows.append({
                "mu": "off (1e6)" if mu >= 1e5 else mu,
                "theta": theta if theta is not None else "off",
                "posterior": post,
                "prior_only": {**summary(max_roots(pdraws)),
                               "prior_df": d0},
                "log_ml": round(float(m.log_marginal_likelihood()), 2),
                "holdout_theta_off": held[mu],
            })
            print(rows[-1]["mu"], rows[-1]["theta"], post, rows[-1]["log_ml"],
                  held[mu]["crps_mean"])

    opt = optimize_hyperparameters(y, lags=LAGS, dummies=D)
    m = BVAR(y, lags=LAGS, dummies=D, lam=opt["lam"], mu=opt["mu"],
             theta=opt["theta"])
    opt_row = {**{key: round(v, 4) for key, v in opt.items()},
               "posterior": summary(max_roots(m.sample_posterior(
                   args.draws, seed=ROOTS_SEED)))}
    print("ML-optimal", opt_row)

    report = {
        "sample": f"{EST_START}-{EST_END}", "lam": 0.2, "lags": LAGS,
        "covid_dummies": "2020Q1-2021Q2", "draws": args.draws,
        "seed": ROOTS_SEED,
        "paper_values": (
            "none stated: Brignone & Piffer (2025) p.10 fn 4 cite GLP (2015) "
            "Minnesota (random walk) + sum-of-coefficients; the repo's "
            "lam 0.2, mu 1, theta 1 are GLP's hyperprior modes"),
        "grid": rows,
        "ml_optimal": opt_row,
        "holdout_note": (
            "held-out columns use the coverage evaluation's convention "
            "(theta off) at each mu, with the six Covid dummies; 49 origins"),
        "code_version": evaluation_code_version(extra_paths=[
            sv.__file__, __file__,
            Path(__file__).with_name("run_common_sv_evaluation.py"),
            Path(__file__).with_name("run_coverage_evaluation.py"),
            Path(__file__).with_name("run_covid_volatility_evaluation.py")]),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()

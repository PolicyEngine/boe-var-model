#!/usr/bin/env python3
"""Rolling-origin pseudo-out-of-sample validation of the unemployment satellite.

Design:

* Expanding-window origins over 2005Q1-2023Q1 (8-quarter final horizon,
  73 origins). At each origin the satellite equation (an
  estimated GDP-growth-to-unemployment mapping: du on growth and lagged
  du, OLS, furlough quarters dummied out) is re-fit on data from
  1992Q1 through the origin only (COVID quarters dummied out per
  satellite.COVID_QUARTERS) -- no future information enters the fit.
* Three GDP inputs:
    - ``realised``: actual future GDP growth. Conditional upper bound --
      "if the GDP forecast were perfect, does the satellite add information?"
    - ``frozen``: GDP growth held at its origin value. Naive input floor.
    - ``svar``: the BVAR re-fit through the origin (posterior-mean,
      unconditional forecast), GDP levels converted to y/y growth. The
      honest end-to-end design.
* Scoring both over all target quarters and excluding COVID targets
  (2020Q1-2021Q2): furlough decoupled unemployment from GDP, so no
  GDP-based mapping can be expected to fit those quarters, and including
  them rewards an attenuated (wrong) growth coefficient.
* Baselines forecast the unemployment rate directly: random walk, drift,
  and the mean-deviation AR(1) from evaluation.py. Diebold-Mariano tests
  vs the random walk (HLN-corrected).
* Band coverage (``band_coverage``), h1-h4 (the published horizon): at each
  origin the same no-dummy BVAR(4) gives BAND_DRAWS reduced-form posterior
  draws, one stochastic GDP path each (``forecast.sample_forecast``, raw,
  not band-factor recalibrated). Three unemployment bands are scored
  against the outturn: ``gdp_only`` (the GDP quantile paths mapped through
  the fitted equation, ``unemployment_bands``, as published before),
  ``gdp_residual`` (plus a N(0, sigma) residual per step) and ``joint``
  (plus coefficient draws; ``unemployment_predictive_bands``).

Writes ``unemployment_satellite_validation.json`` to the results directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boe_var import unemployment_satellite as satellite  # noqa: E402
from boe_var.bvar import BVAR, PosteriorDraw  # noqa: E402
from boe_var.data import load_data, results_path  # noqa: E402
from boe_var.evaluation import (  # noqa: E402
    _ar1_forecast,
    _diebold_mariano,
    _drift_forecast,
    _rmse,
    evaluation_code_version,
)
from boe_var.analysis import aggregate  # noqa: E402
from boe_var.forecast import (  # noqa: E402
    sample_forecast,
    unconditional_forecast,
    yoy,
)

HORIZONS = 8
FIRST_ORIGIN = "2005Q1"
LAST_ORIGIN = "2023Q1"
MODELS = ("realised", "frozen", "svar")
BASELINES = ("rw", "drift", "ar1")
BAND_HORIZONS = 4          # the published unemployment path is capped at 4
BAND_DRAWS = 1000          # posterior draws (one GDP path each) per origin
BAND_SEED = 20261116
BAND_KINDS = ("gdp_only", "gdp_residual", "joint")
LEVELS = {68: ("lo68", "hi68"), 90: ("lo90", "hi90")}


def _qadd(q: str, h: int) -> str:
    year, qq = int(q[:4]), int(q[-1])
    qq += h
    year += (qq - 1) // 4
    qq = (qq - 1) % 4 + 1
    return f"{year}Q{qq}"


def main() -> None:
    df = load_data()
    quarters = [str(q) for q in df.index]
    Y = df.to_numpy()
    k = Y.shape[1]
    gdp_col = list(df.columns).index("uk_gdp")
    g = yoy(Y[:, gdp_col])
    g_quarters = quarters[4:]
    gq_index = {q: j for j, q in enumerate(g_quarters)}

    uq, u = satellite.load_unemployment()
    u_index = {q: i for i, q in enumerate(uq)}

    keys = MODELS + BASELINES
    err = {kk: [[] for _ in range(HORIZONS)] for kk in keys}
    target_is_covid = [[] for _ in range(HORIZONS)]
    # inside[kind][level][h] -> list of bools over origins
    inside = {kind: {lv: [[] for _ in range(BAND_HORIZONS)] for lv in LEVELS}
              for kind in BAND_KINDS}
    half68 = {kind: [[] for _ in range(BAND_HORIZONS)] for kind in BAND_KINDS}
    half68["_covid"] = []

    used = 0
    for origin in quarters:
        if not (FIRST_ORIGIN <= origin <= LAST_ORIGIN):
            continue
        t0 = quarters.index(origin)
        j0 = gq_index[origin]
        i0 = u_index.get(origin)
        if i0 is None or i0 + HORIZONS >= len(u) or t0 + HORIZONS >= len(Y):
            continue

        fit = satellite.fit_default(g_quarters[: j0 + 1], g[: j0 + 1],
                               start="1992Q1", end=origin)
        u_last, du_last = u[i0], u[i0] - u[i0 - 1]

        train = Y[: t0 + 1]
        model = BVAR(train, lags=4)
        sigma = model.S_post / max(model.df_post - k - 1, 1)
        draw = PosteriorDraw(Pi=model.B_post.T.copy(), Sigma=sigma,
                             lags=4, k=k)
        fc_levels = unconditional_forecast(draw, train, horizons=HORIZONS)
        levels = np.r_[train[:, gdp_col], fc_levels[:, gdp_col]]
        g_svar = np.array([levels[t0 + 1 + h] - levels[t0 + 1 + h - 4]
                           for h in range(HORIZONS)])

        paths = {
            "realised": fit.forecast(g[j0 + 1: j0 + 1 + HORIZONS],
                                     u_last, du_last),
            "frozen": fit.forecast(np.full(HORIZONS, g[j0]), u_last, du_last),
            "svar": fit.forecast(g_svar, u_last, du_last),
        }
        # Predictive bands from simulated GDP paths (see module docstring).
        # Separate streams for the posterior draws and for the shock /
        # residual / coefficient draws.
        draws = model.sample_posterior(BAND_DRAWS, seed=BAND_SEED + t0)
        rng = np.random.default_rng([BAND_SEED, t0, 1])
        gdp_tail = train[-4:, gdp_col]
        g_paths = np.empty((BAND_DRAWS, BAND_HORIZONS))
        for n_, d_ in enumerate(draws):
            lv_ = sample_forecast(d_, train, horizons=BAND_HORIZONS,
                                  rng=rng)[:, gdp_col]
            ext = np.r_[gdp_tail, lv_]
            g_paths[n_] = ext[4:] - ext[:-4]
        gdp_q = aggregate(list(g_paths))
        bands = {
            "gdp_only": satellite.unemployment_bands(
                fit, gdp_q, u_last, du_last),
            "gdp_residual": satellite.unemployment_predictive_bands(
                fit, g_paths, u_last, du_last, rng, coefficients=False),
            "joint": satellite.unemployment_predictive_bands(
                fit, g_paths, u_last, du_last, rng),
        }
        actual_b = u[i0 + 1: i0 + 1 + BAND_HORIZONS]
        for kind, b in bands.items():
            for h in range(BAND_HORIZONS):
                for lv, (lo, hi) in LEVELS.items():
                    inside[kind][lv][h].append(
                        bool(b[lo][h] <= actual_b[h] <= b[hi][h]))
                half68[kind][h].append(0.5 * (b["hi68"][h] - b["lo68"][h]))
        half68_covid = [_qadd(origin, h + 1) in satellite.COVID_QUARTERS
                        for h in range(BAND_HORIZONS)]
        half68["_covid"].append(half68_covid)

        u_train = u[max(0, i0 - 60): i0 + 1][:, None]
        base = {"rw": np.full(HORIZONS, u_last),
                "drift": _drift_forecast(u_train, HORIZONS)[:, 0],
                "ar1": _ar1_forecast(u_train, HORIZONS)[:, 0]}

        actual = u[i0 + 1: i0 + 1 + HORIZONS]
        for h in range(HORIZONS):
            for kk in MODELS:
                err[kk][h].append(paths[kk][h] - actual[h])
            for kk in BASELINES:
                err[kk][h].append(base[kk][h] - actual[h])
            target_is_covid[h].append(
                _qadd(origin, h + 1) in satellite.COVID_QUARTERS)
        used += 1

    def score(mask_covid: bool) -> list[dict]:
        rows = []
        for h in range(HORIZONS):
            keep = np.array([not (mask_covid and c)
                             for c in target_is_covid[h]])
            e = {kk: np.asarray(err[kk][h])[keep][:, None] for kk in keys}
            rmse = {kk: float(_rmse(v)[0]) for kk, v in e.items()}
            row = {"horizon": h + 1, "n": int(keep.sum()), "rmse": rmse,
                   "ratio_vs_rw": {kk: rmse[kk] / rmse["rw"]
                                   for kk in keys if kk != "rw"}}
            for kk in MODELS:
                stat, p = _diebold_mariano(e[kk], e["rw"], h + 1)
                row[f"dm_{kk}_vs_rw"] = {"stat": float(stat[0]),
                                         "p": float(p[0])}
            rows.append(row)
        return rows

    def coverage(mask_covid: bool) -> dict:
        out = {}
        for kind in BAND_KINDS:
            out[kind] = {}
            for lv in LEVELS:
                row = {}
                for h in range(BAND_HORIZONS):
                    keep = np.array([not (mask_covid and c)
                                     for c in target_is_covid[h]])
                    hits = np.asarray(inside[kind][lv][h])[keep]
                    row[f"h{h + 1}"] = {"coverage": round(float(hits.mean()), 4),
                                        "n": int(keep.sum())}
                out[kind][str(lv)] = row
            cov_ = np.asarray(half68["_covid"], dtype=bool)
            out[kind]["mean_half_width_68"] = {
                f"h{h + 1}": round(float(np.mean(
                    np.asarray(half68[kind][h])[
                        ~cov_[:, h] if mask_covid else slice(None)])), 4)
                for h in range(BAND_HORIZONS)}
        return out

    results = {
        "design": {
            "first_origin": FIRST_ORIGIN, "last_origin": LAST_ORIGIN,
            "origins_used": used, "horizons": HORIZONS,
            "fit_sample_start": "1992Q1",
            "covid_quarters_dummied": sorted(satellite.COVID_QUARTERS),
            "gdp_inputs": list(MODELS),
            "note": "svar = posterior-mean BVAR re-fit per origin "
                    "(end-to-end). See band_coverage for the measured "
                    "coverage of GDP-only and joint predictive bands.",
        },
        "band_coverage": {
            "method": (
                f"h1-h{BAND_HORIZONS}; per origin {BAND_DRAWS} reduced-form "
                "posterior draws of the no-dummy BVAR(4) refit through the "
                "origin, one stochastic GDP path each (raw, not recalibrated); "
                "gdp_only maps the GDP quantile paths through the fitted "
                "equation (the bands published before); gdp_residual adds a "
                "N(0, sigma) residual per step; joint also draws the "
                "coefficients from N(coef, sigma^2 (X'X)^-1). Residual and "
                "coefficient draws are independent of the GDP shock path. "
                f"Posterior draws seeded {BAND_SEED} + origin index; shock, "
                f"residual and coefficient draws from a separate stream "
                f"[{BAND_SEED}, origin index, 1]."
            ),
            "caveat": (
                "73 overlapping origins, 67 of them with a non-Covid target "
                "at each horizon: binomial s.e. about 0.037 at 90% and 0.057 "
                "at 68% (n = 67) before serial correlation, so the true "
                "uncertainty is larger. The GDP paths come from a no-dummy "
                "BVAR refit per origin, not the production specification, and "
                "raw SVAR GDP bands themselves under-cover. Assumes a stable "
                "GDP-unemployment relation and a constant Gaussian residual "
                "variance (Covid quarters excluded from the fit). Latest-"
                "vintage data (pseudo-, not real-time)."
            ),
            "ex_covid_targets": coverage(mask_covid=True),
            "all_targets": coverage(mask_covid=False),
        },
        "all_targets": score(mask_covid=False),
        "ex_covid_targets": score(mask_covid=True),
        # Staleness guard: hash of the evaluation-relevant sources plus the
        # satellite module and this script. tests/test_committed_artifacts.py
        # fails if the code changes without this artifact being regenerated.
        "code_version": evaluation_code_version(
            extra_paths=[satellite.__file__, satellite._MGSX_PATH, __file__]
        ),
    }

    out = results_path("unemployment_satellite_validation.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out} ({used} origins)")
    bc = results["band_coverage"]["ex_covid_targets"]
    print("--- band coverage, ex-Covid targets (68% / 90%)")
    for kind in BAND_KINDS:
        print(kind, " ".join(
            f"h{h}: {bc[kind]['68'][f'h{h}']['coverage']:.2f}/"
            f"{bc[kind]['90'][f'h{h}']['coverage']:.2f}"
            for h in range(1, BAND_HORIZONS + 1)),
            "half68", bc[kind]["mean_half_width_68"])
    for label in ("all_targets", "ex_covid_targets"):
        print(f"--- {label}")
        print("h  realised/rw  frozen/rw  svar/rw  ar1/rw  DM-p(svar)")
        for r in results[label]:
            q = r["ratio_vs_rw"]
            print(f"{r['horizon']}  {q['realised']:.3f}        "
                  f"{q['frozen']:.3f}      {q['svar']:.3f}    "
                  f"{q['ar1']:.3f}   {r['dm_svar_vs_rw']['p']:.3f}")


if __name__ == "__main__":
    main()

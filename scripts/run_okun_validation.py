#!/usr/bin/env python3
"""Rolling-origin pseudo-out-of-sample validation of the Okun satellite.

Design:

* Expanding-window origins over 2005Q1-2023Q1 (8-quarter final horizon,
  73 origins). At each origin the Okun equation is re-fit on data from
  1992Q1 through the origin only (COVID quarters dummied out per
  okun.COVID_QUARTERS) -- no future information enters the fit.
* Three GDP inputs:
    - ``realised``: actual future GDP growth. Conditional upper bound --
      "if the GDP forecast were perfect, does Okun add information?"
    - ``frozen``: GDP growth held at its origin value. Naive input floor.
    - ``svar``: the BVAR re-fit through the origin (posterior-mean,
      unconditional forecast), GDP levels converted to y/y growth. The
      honest end-to-end design.
* Scoring both over all target quarters and excluding COVID targets
  (2020Q1-2021Q2): furlough decoupled unemployment from GDP, so no
  GDP-based mapping can be expected to fit those quarters, and including
  them rewards an attenuated (wrong) Okun coefficient.
* Baselines forecast the unemployment rate directly: random walk, drift,
  and the mean-deviation AR(1) from evaluation.py. Diebold-Mariano tests
  vs the random walk (HLN-corrected).

Writes ``okun_validation.json`` to the results directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boe_var import okun  # noqa: E402
from boe_var.bvar import BVAR, PosteriorDraw  # noqa: E402
from boe_var.data import load_data, results_path  # noqa: E402
from boe_var.evaluation import (  # noqa: E402
    _ar1_forecast,
    _diebold_mariano,
    _drift_forecast,
    _rmse,
)
from boe_var.forecast import unconditional_forecast, yoy  # noqa: E402

HORIZONS = 8
FIRST_ORIGIN = "2005Q1"
LAST_ORIGIN = "2023Q1"
MODELS = ("realised", "frozen", "svar")
BASELINES = ("rw", "drift", "ar1")


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

    uq, u = okun.load_unemployment()
    u_index = {q: i for i, q in enumerate(uq)}

    keys = MODELS + BASELINES
    err = {kk: [[] for _ in range(HORIZONS)] for kk in keys}
    target_is_covid = [[] for _ in range(HORIZONS)]

    used = 0
    for origin in quarters:
        if not (FIRST_ORIGIN <= origin <= LAST_ORIGIN):
            continue
        t0 = quarters.index(origin)
        j0 = gq_index[origin]
        i0 = u_index.get(origin)
        if i0 is None or i0 + HORIZONS >= len(u) or t0 + HORIZONS >= len(Y):
            continue

        fit = okun.fit_default(g_quarters[: j0 + 1], g[: j0 + 1],
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
                _qadd(origin, h + 1) in okun.COVID_QUARTERS)
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

    results = {
        "design": {
            "first_origin": FIRST_ORIGIN, "last_origin": LAST_ORIGIN,
            "origins_used": used, "horizons": HORIZONS,
            "fit_sample_start": "1992Q1",
            "covid_quarters_dummied": sorted(okun.COVID_QUARTERS),
            "gdp_inputs": list(MODELS),
            "note": "svar = posterior-mean BVAR re-fit per origin "
                    "(end-to-end). Bands in production carry GDP "
                    "uncertainty only; Okun residual uncertainty is not "
                    "sampled.",
        },
        "all_targets": score(mask_covid=False),
        "ex_covid_targets": score(mask_covid=True),
    }

    out = results_path("okun_validation.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out} ({used} origins)")
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

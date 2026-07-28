#!/usr/bin/env python3
"""Rolling-origin pseudo-out-of-sample validation of the Okun satellite.

Design (documented limits):

* Expanding-window origins over 2005Q1-2023Q1 (8-quarter final horizon).
* At each origin the Okun equation is re-fit on data from 1992Q1 through the
  origin only -- no future information enters the fit.
* Two GDP inputs bracket real-time skill WITHOUT re-fitting the BVAR at every
  origin (that full exercise is the follow-up before any published claim of
  end-to-end real-time skill):
    - ``realised``: actual future GDP growth. Conditional upper bound --
      "if the GDP forecast were perfect, does Okun add information about
      unemployment?"
    - ``frozen``: GDP growth held at its origin value. A feasible naive
      real-time input; a lower bound on what the satellite + any GDP
      forecast with skill would deliver.
* Baselines forecast the unemployment rate directly: random walk (no
  change), drift, and the mean-deviation AR(1) from evaluation.py.
* RMSE ratios (model/baseline, <1 favours the satellite) and Diebold-Mariano
  p-values vs the random walk per horizon 1-8.

Writes ``okun_validation.json`` to the results directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boe_var import okun  # noqa: E402
from boe_var.data import load_data, results_path  # noqa: E402
from boe_var.evaluation import (  # noqa: E402
    _ar1_forecast,
    _diebold_mariano,
    _drift_forecast,
    _rmse,
)
from boe_var.forecast import yoy  # noqa: E402

HORIZONS = 8
FIRST_ORIGIN = "2005Q1"
LAST_ORIGIN = "2023Q1"


def main() -> None:
    df = load_data()
    quarters = [str(q) for q in df.index]
    gdp_levels = np.asarray(df["uk_gdp"], dtype=float)
    g = yoy(gdp_levels)                 # percent y/y, starts at quarters[4]
    g_quarters = quarters[4:]

    uq, u = okun.load_unemployment()
    u_index = {q: i for i, q in enumerate(uq)}
    gq_index = {q: j for j, q in enumerate(g_quarters)}

    origins = [q for q in g_quarters if FIRST_ORIGIN <= q <= LAST_ORIGIN]
    err = {"realised": [[] for _ in range(HORIZONS)],
           "frozen": [[] for _ in range(HORIZONS)],
           "rw": [[] for _ in range(HORIZONS)],
           "drift": [[] for _ in range(HORIZONS)],
           "ar1": [[] for _ in range(HORIZONS)]}

    used = 0
    for origin in origins:
        j0, i0 = gq_index[origin], u_index.get(origin)
        if i0 is None or i0 + HORIZONS >= len(u) or j0 + HORIZONS >= len(g):
            continue
        fit = okun.fit_default(g_quarters[: j0 + 1], g[: j0 + 1],
                               start="1992Q1", end=origin)
        u_last, du_last = u[i0], u[i0] - u[i0 - 1]
        actual = u[i0 + 1: i0 + 1 + HORIZONS]

        path_real = fit.forecast(g[j0 + 1: j0 + 1 + HORIZONS], u_last, du_last)
        path_frozen = fit.forecast(np.full(HORIZONS, g[j0]), u_last, du_last)

        train = u[max(0, i0 - 60): i0 + 1][:, None]
        drift = _drift_forecast(train, HORIZONS)[:, 0]
        ar1 = _ar1_forecast(train, HORIZONS)[:, 0]

        for h in range(HORIZONS):
            err["realised"][h].append(path_real[h] - actual[h])
            err["frozen"][h].append(path_frozen[h] - actual[h])
            err["rw"][h].append(u_last - actual[h])
            err["drift"][h].append(drift[h] - actual[h])
            err["ar1"][h].append(ar1[h] - actual[h])
        used += 1

    results = {"design": {"first_origin": FIRST_ORIGIN,
                          "last_origin": LAST_ORIGIN, "origins_used": used,
                          "horizons": HORIZONS,
                          "fit_sample_start": "1992Q1",
                          "gdp_inputs": ["realised (upper bound)",
                                         "frozen (naive real-time input)"],
                          "note": "No BVAR re-fit per origin; end-to-end "
                                  "real-time skill is bracketed, not "
                                  "measured. Bands carry GDP uncertainty "
                                  "only."},
               "per_horizon": []}
    for h in range(HORIZONS):
        e = {k: np.asarray(v[h])[:, None] for k, v in err.items()}
        rmse = {k: float(_rmse(v)[0]) for k, v in e.items()}
        row = {"horizon": h + 1, "rmse": rmse,
               "ratio_vs_rw": {k: rmse[k] / rmse["rw"]
                               for k in ("realised", "frozen",
                                         "drift", "ar1")}}
        for k in ("realised", "frozen"):
            stat, p = _diebold_mariano(e[k], e["rw"], h + 1)
            row[f"dm_{k}_vs_rw"] = {"stat": float(stat[0]),
                                    "p": float(p[0])}
        results["per_horizon"].append(row)

    out = results_path("okun_validation.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out} ({used} origins)")
    hdr = "h  realised/rw  frozen/rw  ar1/rw  drift/rw   DM-p(real)  DM-p(frozen)"
    print(hdr)
    for r in results["per_horizon"]:
        q = r["ratio_vs_rw"]
        print(f"{r['horizon']}  {q['realised']:.3f}        {q['frozen']:.3f}"
              f"      {q['ar1']:.3f}   {q['drift']:.3f}     "
              f"{r['dm_realised_vs_rw']['p']:.3f}       "
              f"{r['dm_frozen_vs_rw']['p']:.3f}")


if __name__ == "__main__":
    main()

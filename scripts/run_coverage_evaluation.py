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
# Predictive quantile levels recorded per origin for the full-distribution
# recalibration and the CRPS (symmetric, so every level below the median has a
# central-interval partner above it; 0.16/0.84 and 0.05/0.95 are the published
# 68% and 90% band edges).
TAUS = (0.025, 0.05, 0.10, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.84, 0.90, 0.95, 0.975)
CONFORMAL_MIN_TRAIN = 12


def _z(p: float) -> float:
    """Inverse standard-normal CDF by bisection on erf (no scipy import)."""
    from math import erf, sqrt

    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + erf(mid / sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _scale_factor(nominal: int, empirical: float) -> float | None:
    """k = z(nominal) / z(empirical); None when empirical is 0 or 1."""
    if not 0.0 < empirical < 1.0:
        return None
    return _z((1.0 + nominal / 100.0) / 2.0) / _z((1.0 + empirical) / 2.0)


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
    record: dict | None = None,
) -> tuple[dict, np.ndarray]:
    """Empirical band coverage. Returns (hits-per-level, origin count per h).

    ``record``: if a dict is passed, it is filled with the per-origin band
    geometry the held-out calibration needs -- ``stretch[level]`` of shape
    (origins, horizons, k), the factor by which that origin's band
    half-width (on the side the outturn fell) would have to be scaled for
    the outturn to sit on the band edge. The outturn is inside the band
    scaled by k exactly when stretch <= k, so stretch <= 1 reproduces
    ``hits``.

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
    stretch = {lv: [] for lv in (68, 90)}
    quant, acts, pit = [], [], []

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
        # full predictive quantile grid (origins, horizons, k, len(TAUS)),
        # the outturns (origins, horizons, k) and the ensemble PIT, for the
        # conformal recalibration of the whole distribution (PIT, CRPS)
        record["quantiles"] = np.asarray(quant)
        record["actual"] = np.asarray(acts)
        record["pit"] = np.asarray(pit)
    return hits, n


def band_stretch(med, lo, hi, actual) -> np.ndarray:
    """Half-width multiple at which ``actual`` sits on the band edge.

    Uses the upper half-width when the outturn is above the median and the
    lower one otherwise, the same around-the-median rescaling the published
    fans apply. A zero half-width gives inf (0 if the outturn is the median).
    """
    med, lo, hi, actual = (np.asarray(x, dtype=float)
                           for x in (med, lo, hi, actual))
    up = actual >= med
    dist = np.where(up, actual - med, med - actual)
    half = np.where(up, hi - med, med - lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(half > 0, dist / np.where(half > 0, half, 1.0),
                       np.where(dist > 0, np.inf, 0.0))
    return out


def holdout_calibration(stretch: dict, min_train: int = 12) -> dict:
    """Coverage of the k-calibrated bands, scored on origins k was NOT fitted on.

    ``stretch``: {level: (origins, horizons, k)} from ``run_coverage(record=)``.
    For each level, horizon and variable, k = z(nominal) / z(empirical) as in
    ``band_scale_factors``, then a band rescaled by k covers an outturn iff
    stretch <= k. Three schemes:

    - ``in_sample``: k fitted on all origins and judged on the same origins,
      which is what the committed factors amount to;
    - ``leave_one_out``: k refitted without the origin being scored;
    - ``rolling_origin``: k fitted only on origins whose h-step outturn was
      already observed at the scored origin (origin j with j + h <= i), the
      only scheme a forecaster could have run in real time. The first
      ``min_train`` usable origins per horizon are not scored.

    Where the training coverage is 0 or 1 there is no finite k and the band
    is left unscaled (k = 1). Returns means across variables per horizon,
    per variable across horizons, and overall, plus the raw (k = 1) coverage
    on the same scored origins for comparison.
    """
    out = {}
    for lv, S in stretch.items():
        S = np.asarray(S, dtype=float)
        n_o, H, k = S.shape
        inside = S <= 1.0
        res = {name: np.full((n_o, H, k), np.nan)
               for name in ("in_sample", "leave_one_out", "rolling_origin",
                            "raw")}
        res["raw"][:] = inside
        for h in range(H):
            for j in range(k):
                col_in = inside[:, h, j]
                k_all = _scale_factor(lv, float(col_in.mean()))
                res["in_sample"][:, h, j] = S[:, h, j] <= (k_all or 1.0)
                total = col_in.sum()
                for i in range(n_o):
                    emp = (total - col_in[i]) / (n_o - 1)
                    kf = _scale_factor(lv, float(emp)) or 1.0
                    res["leave_one_out"][i, h, j] = S[i, h, j] <= kf
                    n_train = i - (h + 1) + 1   # origins 0..i-h-1
                    if n_train >= min_train:
                        emp_r = float(col_in[:n_train].mean())
                        kr = _scale_factor(lv, emp_r) or 1.0
                        res["rolling_origin"][i, h, j] = S[i, h, j] <= kr
        rolled = ~np.isnan(res["rolling_origin"])
        raw_rolled = np.where(rolled, res["raw"], np.nan)

        def summ(a):
            return {
                "mean": round(float(np.nanmean(a)), 4),
                "by_horizon": {f"h{h + 1}": round(float(np.nanmean(a[:, h])), 4)
                               for h in range(H)},
                "by_variable": {var: round(float(np.nanmean(a[:, :, j])), 4)
                                for j, var in enumerate(COLUMNS[:k])},
            }

        out[str(lv)] = {
            "nominal": lv / 100.0,
            "raw": summ(res["raw"]),
            "in_sample_calibrated": summ(res["in_sample"]),
            "leave_one_out_calibrated": summ(res["leave_one_out"]),
            "rolling_origin_calibrated": summ(res["rolling_origin"]),
            "rolling_origin_raw_same_origins": summ(raw_rolled),
            "rolling_origin_scored_origins_by_horizon": {
                f"h{h + 1}": int(rolled[:, h, 0].sum()) for h in range(H)},
        }
    return out


def conformal_factor(scores, level: float) -> tuple[float, bool]:
    """Split-conformal half-width factor from past band stretches.

    ``scores`` are past outturns' stretches (``band_stretch``) for one level,
    horizon and variable; ``level`` is the nominal coverage (0.9 for the 90%
    band). Returns the ``ceil((n + 1) * level)``-th smallest score -- the
    finite-sample conformal quantile: if the scored origin's stretch is
    exchangeable with the past ones, a band scaled by it covers with
    probability >= ``level``. When that rank exceeds ``n`` (too few past
    origins for the level) the largest score is used instead and the second
    return value is True, so the shortfall is recorded, not hidden.

    Distribution-free: unlike ``z(nominal) / z(empirical)`` it does not
    assume the predictive errors are Gaussian, so it can widen the 90% band
    by more than the 68% band when the tails are fat.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    if n == 0:
        raise ValueError("no calibration scores")
    rank = int(np.ceil((n + 1) * level))
    if rank > n:
        return float(s[-1]), True
    return float(s[rank - 1]), False


def conformal_rolling_factors(S, level: float, *, min_train: int =
                              CONFORMAL_MIN_TRAIN, exclude=None):
    """Real-time conformal factor for every origin, horizon and variable.

    ``S``: stretches (origins, horizons, k) for one band level. The factor
    for origin ``i`` at horizon ``h`` (0-based) uses only origins
    ``j <= i - h - 1`` -- those whose (h+1)-step outturn has been observed by
    origin ``i`` -- so nothing at or after the scored origin enters its own
    band. ``exclude`` (origins, horizons) bool drops calibration origins (for
    example outturns in the Covid quarters) from the training set; they are
    still scored. Origins with fewer than ``min_train`` usable training
    origins get NaN (not scored). Returns (factor, capped).
    """
    S = np.asarray(S, dtype=float)
    n_o, H, k = S.shape
    ex = (np.zeros((n_o, H), bool) if exclude is None
          else np.asarray(exclude, bool))
    K = np.full((n_o, H, k), np.nan)
    capped = np.zeros((n_o, H, k), bool)
    for h in range(H):
        for i in range(n_o):
            n_train = i - h          # origins 0 .. i - h - 1
            if n_train <= 0:
                continue
            use = ~ex[:n_train, h]
            if use.sum() < min_train:
                continue
            for j in range(k):
                K[i, h, j], capped[i, h, j] = conformal_factor(
                    S[:n_train, h, j][use], level)
    return K, capped


def gaussian_rolling_factors(S, nominal: int, *, min_train: int =
                             CONFORMAL_MIN_TRAIN) -> np.ndarray:
    """The rolling-origin ``z(nominal)/z(empirical)`` factors scored in
    ``holdout_calibration`` (k = 1 where empirical is 0 or 1), as an array,
    so their widths can be set against the conformal ones origin by origin."""
    S = np.asarray(S, dtype=float)
    n_o, H, k = S.shape
    inside = S <= 1.0
    K = np.full((n_o, H, k), np.nan)
    for h in range(H):
        for i in range(n_o):
            n_train = i - h
            if n_train < min_train:
                continue
            for j in range(k):
                emp = float(inside[:n_train, h, j].mean())
                K[i, h, j] = _scale_factor(nominal, emp) or 1.0
    return K


def _central_pairs(taus):
    """(level, lower index, upper index, median index) for each central
    interval in a symmetric quantile grid."""
    taus = np.round(np.asarray(taus, dtype=float), 6)
    mid = int(np.flatnonzero(taus == 0.5)[0])
    out = []
    for a in range(mid):
        b = int(np.flatnonzero(np.isclose(taus, 1.0 - taus[a]))[0])
        out.append((round(1.0 - 2.0 * taus[a], 6), a, b, mid))
    return out


def recalibrated_quantiles(Q, actual, taus=TAUS, *, min_train: int =
                           CONFORMAL_MIN_TRAIN, exclude=None):
    """Conformal recalibration of the whole predictive distribution.

    ``Q``: predictive quantiles (origins, horizons, k, len(taus)); ``actual``:
    outturns (origins, horizons, k). For each central level L in the grid the
    L-band is rescaled around the median by its own rolling conformal factor
    k_L, i.e. ``Q'_tau = median + k_L * (Q_tau - median)`` with
    ``L = |2 tau - 1|``, then sorted along tau so it stays a quantile
    function. Origins without a full training window are NaN. Returns
    (Q', factors {L: array}).
    """
    Q = np.asarray(Q, dtype=float)
    actual = np.asarray(actual, dtype=float)
    Qr = np.full_like(Q, np.nan)
    med = Q[..., _central_pairs(taus)[0][3]]
    Qr[..., _central_pairs(taus)[0][3]] = med
    factors = {}
    for L, a, b, m in _central_pairs(taus):
        S = band_stretch(med, Q[..., a], Q[..., b], actual)
        K, _ = conformal_rolling_factors(S, L, min_train=min_train,
                                         exclude=exclude)
        factors[L] = K
        Qr[..., a] = med + K * (Q[..., a] - med)
        Qr[..., b] = med + K * (Q[..., b] - med)
    ok = ~np.isnan(Qr).any(axis=-1)
    Qr[ok] = np.sort(Qr[ok], axis=-1)
    Qr[~ok] = np.nan
    return Qr, factors


def grid_pit(Q, actual, taus=TAUS) -> np.ndarray:
    """PIT of the outturn under a quantile-grid predictive (linear between
    grid points; below the lowest / above the highest grid level it is set to
    the midpoint of the uncovered tail, e.g. 0.0125 for a 2.5% bottom tail)."""
    Q = np.asarray(Q, dtype=float)
    actual = np.asarray(actual, dtype=float)
    taus = np.asarray(taus, dtype=float)
    out = np.full(actual.shape, np.nan)
    flatQ = Q.reshape(-1, Q.shape[-1])
    flaty = actual.reshape(-1)
    res = out.reshape(-1)
    for n, (q, y) in enumerate(zip(flatQ, flaty)):
        if np.isnan(q).any() or np.isnan(y):
            continue
        if y < q[0]:
            res[n] = taus[0] / 2.0
        elif y > q[-1]:
            res[n] = (1.0 + taus[-1]) / 2.0
        else:
            res[n] = np.interp(y, q, taus)
    return res.reshape(actual.shape)


def quantile_crps(Q, actual, taus=TAUS) -> np.ndarray:
    """CRPS approximated by the quantile score integrated over the grid,
    CRPS = int_0^1 2 (1{y < q_tau} - tau)(q_tau - y) d tau, by the trapezoid
    rule on ``taus`` (so the outer 2.5% tails are left out; the same
    truncation applies to every method compared). Lower is better; in model
    units, so compare within a variable."""
    Q = np.asarray(Q, dtype=float)
    y = np.asarray(actual, dtype=float)[..., None]
    taus = np.asarray(taus, dtype=float)
    qs = 2.0 * ((y < Q).astype(float) - taus) * (Q - y)
    w = np.diff(taus)
    return (0.5 * (qs[..., 1:] + qs[..., :-1]) * w).sum(axis=-1)


def conformal_evaluation(record: dict, *, min_train: int = CONFORMAL_MIN_TRAIN,
                         exclude=None) -> dict:
    """Rolling-origin (real-time) conformal recalibration scored held out.

    Coverage of the 68% and 90% bands, band width relative to raw and to the
    Gaussian ``z(nominal)/z(empirical)`` factor on the SAME scored origins,
    and PIT histogram and CRPS of the whole distribution before and after.
    """
    S68, S90 = record["stretch"][68], record["stretch"][90]
    n_o, H, k = S68.shape
    Q, y = record["quantiles"], record["actual"]
    Qr, _ = recalibrated_quantiles(Q, y, min_train=min_train, exclude=exclude)
    out = {}
    for lv, S in ((68, S68), (90, S90)):
        K, capped = conformal_rolling_factors(S, lv / 100.0,
                                              min_train=min_train,
                                              exclude=exclude)
        G = gaussian_rolling_factors(S, lv, min_train=min_train)
        scored = ~np.isnan(K)
        cov_c = np.where(scored, S <= K, np.nan)
        cov_g = np.where(scored & ~np.isnan(G), S <= G, np.nan)
        cov_r = np.where(scored, S <= 1.0, np.nan)

        def by(a, f=np.nanmean, nd=4):
            return {
                "mean": round(float(f(a)), nd),
                "by_horizon": {f"h{h + 1}": round(float(f(a[:, h])), nd)
                               for h in range(H)},
                "by_variable": {v: round(float(f(a[:, :, j])), nd)
                                for j, v in enumerate(COLUMNS[:k])},
                "by_variable_horizon": {
                    v: {f"h{h + 1}": round(float(f(a[:, h, j])), nd)
                        for h in range(H)}
                    for j, v in enumerate(COLUMNS[:k])},
            }

        Kn = np.where(scored, K, np.nan)
        Gn = np.where(scored, G, np.nan)
        out[str(lv)] = {
            "nominal": lv / 100.0,
            "conformal": by(cov_c),
            "gaussian_k": by(cov_g),
            "raw": by(cov_r),
            "half_width_factor_conformal": by(Kn, nd=3),
            "half_width_factor_gaussian": by(Gn, nd=3),
            "half_width_factor_conformal_max": round(float(np.nanmax(Kn)), 3),
            "conformal_over_gaussian_width": round(
                float(np.nanmean(Kn) / np.nanmean(Gn)), 3),
            "capped_share": round(float(capped[scored].mean()), 4),
            "scored_origins_by_horizon": {
                f"h{h + 1}": int(scored[:, h, 0].sum()) for h in range(H)},
        }
    ok = ~np.isnan(Qr).any(axis=-1)
    pit_raw = np.where(ok, grid_pit(Q, y), np.nan)
    pit_new = np.where(ok, grid_pit(Qr, y), np.nan)
    crps_raw = np.where(ok, quantile_crps(Q, y), np.nan)
    crps_new = np.where(ok, quantile_crps(np.where(ok[..., None], Qr, 0.0), y),
                        np.nan)
    edges = np.linspace(0.0, 1.0, 11)

    def hist(p):
        c, _ = np.histogram(p[~np.isnan(p)], bins=edges)
        return [round(float(x), 4) for x in c / c.sum()]

    out["pit_histogram"] = {
        "bins": [round(float(e), 1) for e in edges],
        "note": "share of scored outturns per PIT decile; calibrated = 0.1 "
                "each. Pooled over variables and horizons.",
        "raw": hist(pit_raw),
        "conformal": hist(pit_new),
        "raw_by_horizon": {f"h{h + 1}": hist(pit_raw[:, h]) for h in (0, 3, 7)
                           if h < H},
        "conformal_by_horizon": {f"h{h + 1}": hist(pit_new[:, h])
                                 for h in (0, 3, 7) if h < H},
        "outside_grid_share_raw": round(float(np.nanmean(
            (pit_raw < TAUS[0]) | (pit_raw > TAUS[-1]))), 4),
        "outside_grid_share_conformal": round(float(np.nanmean(
            (pit_new < TAUS[0]) | (pit_new > TAUS[-1]))), 4),
    }
    out["crps"] = {
        "note": "quantile-score CRPS over tau in [0.025, 0.975], mean over "
                "scored origins and horizons, model units; ratio < 1 means "
                "the recalibrated distribution scores better",
        "raw": {v: round(float(np.nanmean(crps_raw[:, :, j])), 4)
                for j, v in enumerate(COLUMNS[:k])},
        "conformal": {v: round(float(np.nanmean(crps_new[:, :, j])), 4)
                      for j, v in enumerate(COLUMNS[:k])},
        "ratio_by_variable": {
            v: round(float(np.nanmean(crps_new[:, :, j])
                           / np.nanmean(crps_raw[:, :, j])), 3)
            for j, v in enumerate(COLUMNS[:k])},
        "ratio_by_horizon": {
            f"h{h + 1}": round(float(np.nanmean(
                crps_new[:, h] / np.nanmean(crps_raw[:, h], axis=0))), 3)
            for h in range(H)},
    }
    return out


def conformal_production_factors(stretch: dict, exclude=None) -> dict:
    """Factors to apply to the NEXT published fan: the conformal quantile of
    every origin's stretch (each origin's outturn is observed by the last
    data quarter), per level, horizon and variable."""
    out = {}
    for lv, S in stretch.items():
        S = np.asarray(S, dtype=float)
        n_o, H, k = S.shape
        ex = (np.zeros((n_o, H), bool) if exclude is None
              else np.asarray(exclude, bool))
        out[str(lv)] = {}
        for h in range(H):
            out[str(lv)][f"h{h + 1}"] = {}
            for j, v in enumerate(COLUMNS[:k]):
                f, cap = conformal_factor(S[~ex[:, h], h, j], lv / 100.0)
                out[str(lv)][f"h{h + 1}"][v] = round(f, 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lags", type=int, default=LAGS)
    parser.add_argument("--horizons", type=int, default=HORIZONS)
    parser.add_argument("--first-origin", type=int, default=FIRST_ORIGIN)
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--paths", type=int, default=N_PATHS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default="results/coverage_evaluation.json")
    parser.add_argument("--dump-record", default=None,
                        help="optional .npz path for the per-origin arrays")
    args = parser.parse_args()

    import pandas as pd

    df = load_data()
    y = df.to_numpy(float)
    T, k = y.shape
    n_origins = T - args.horizons - args.first_origin

    rec_head, rec_prod = {}, {}
    hits, n = run_coverage(
        y,
        lags=args.lags,
        horizons=args.horizons,
        first_origin=args.first_origin,
        n_draws=args.draws,
        n_paths=args.paths,
        seed=args.seed,
        record=rec_head,
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
        record=rec_prod,
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
        # The data sample above is not the evaluation window: coverage is
        # scored only on these origins (and their outturns), which the Covid
        # and 2022 inflation quarters dominate.
        "evaluated_origins": {
            "first": str(df.index[args.first_origin]),
            "last": str(df.index[T - args.horizons - 1]),
            "last_outturn": str(df.index[-1]),
        },
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
    # The factors above are fitted on the same origins their coverage is
    # judged on, so "calibrated coverage = nominal" holds by construction.
    # Score them on origins they were not fitted on.
    report["calibration_holdout"] = {
        "method": (
            "per level/horizon/variable k = z(nominal)/z(empirical) refitted "
            "without the scored origin (leave_one_out) or on origins whose "
            "outturn was already observed (rolling_origin, the real-time "
            "scheme; first 12 usable origins per horizon not scored); the "
            "band half-width is scaled by k around the median and the "
            "outturn scored against it"
        ),
        "caveat": (
            "Consecutive origins share outturn quarters, so leave-one-out "
            "still leaks neighbouring information and is optimistic; "
            "rolling_origin is the honest number. Measured on series levels "
            "in model units, like the factors themselves."
        ),
        "headline": holdout_calibration(rec_head["stretch"]),
        "production_spec": holdout_calibration(rec_prod["stretch"]),
    }
    # Issue #12: the Gaussian factor cannot reach 90% out of sample. Test the
    # distribution-free alternative: the conformal quantile of past
    # stretches, fitted in real time (each scored origin calibrates only on
    # outturns it had seen). The sensitivity drops calibration outturns in
    # the Covid quarters, which the production model dummies out of
    # estimation. Result (6 Nov 2026): it does not fix the 90% band either --
    # the misses cluster in 2019-22 and no calibration on earlier errors can
    # anticipate them -- so it is recorded as evidence, NOT adopted; the
    # published correction stays band_scale_factors.
    n_rec = rec_prod["stretch"][68].shape[0]
    covid_out = np.zeros((n_rec, args.horizons), bool)
    for i in range(n_rec):
        for h in range(args.horizons):
            covid_out[i, h] = df.index[args.first_origin + i + h + 1] in covid_q
    report["conformal_calibration"] = {
        "method": (
            "split-conformal: per level/horizon/variable the band half-width "
            "around the median is scaled by the ceil((n+1)*level)-th smallest "
            "of the n past band stretches (outturn distance from the median "
            "over the half-width on that side); rolling origin, so origin i "
            "uses only origins whose outturn was observed at i; first "
            f"{CONFORMAL_MIN_TRAIN} usable origins per horizon not scored. "
            "The whole distribution is recalibrated the same way level by "
            "level for the PIT histogram and CRPS."
        ),
        "verdict": (
            "not adopted: out of sample the conformal 90% band covers no more "
            "than the Gaussian-k band, at a wider average width and a worse "
            "CRPS. Forecast misses cluster in time (2019-22), which "
            "violates the exchangeability the method relies on; the fix is a "
            "model with time-varying volatility (issue #12 option 1)."
        ),
        "caveat": (
            "With 12-48 past origins the 90% factor is a high order "
            "statistic (the 12th of 12 at the start), so one extreme outturn "
            "sets it until enough later origins arrive. Consecutive origins "
            "share outturn quarters. Levels in model units."
        ),
        "headline": conformal_evaluation(rec_head),
        "production_spec": conformal_evaluation(rec_prod),
        "production_spec_covid_outturns_excluded": conformal_evaluation(
            rec_prod, exclude=covid_out),
        "factors_for_next_forecast": {
            "description": (
                "conformal factors fitted on all origins of the production "
                "spec, for comparison with band_scale_factors; NOT applied "
                "to any published fan (see verdict)"
            ),
            "production_spec": conformal_production_factors(
                rec_prod["stretch"]),
            "production_spec_covid_outturns_excluded":
                conformal_production_factors(rec_prod["stretch"],
                                             exclude=covid_out),
        },
    }
    if args.dump_record:
        np.savez_compressed(
            args.dump_record,
            **{f"{tag}_{key}": v
               for tag, rec in (("head", rec_head), ("prod", rec_prod))
               for key, v in (("quantiles", rec["quantiles"]),
                              ("actual", rec["actual"]),
                              ("pit", rec["pit"]))},
            **{f"{tag}_stretch{lv}": rec["stretch"][lv]
               for tag, rec in (("head", rec_head), ("prod", rec_prod))
               for lv in (68, 90)},
            covid_out=covid_out)
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

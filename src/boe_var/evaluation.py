"""Leakage-safe pseudo-out-of-sample forecast evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .bvar import BVAR, PosteriorDraw
from .forecast import unconditional_forecast

__all__ = ["evaluation_code_version", "rolling_origin_evaluation"]

# Source files whose behaviour the committed evaluation artifacts depend on.
# ``evaluation_code_version`` hashes these (plus any caller-supplied extras,
# e.g. the runner script) so a change to evaluation code without regenerating
# the committed JSONs is caught by the staleness test.
_CODE_VERSION_FILES = ("bvar.py", "data.py", "evaluation.py", "forecast.py")


def evaluation_code_version(extra_paths: tuple | list = ()) -> str:
    """SHA-256 over the evaluation-relevant sources, stable across machines.

    ``extra_paths`` lets a runner script include itself in the hash. Files
    are read as bytes in a fixed order, each prefixed with its basename, so
    renames and reorderings change the digest too.
    """
    here = Path(__file__).resolve().parent
    paths = [here / name for name in _CODE_VERSION_FILES]
    paths += [Path(p).resolve() for p in extra_paths]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _rmse(errors: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(errors), axis=0))


def _diebold_mariano(
    model_err: np.ndarray, rw_err: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Diebold-Mariano test of equal squared-error accuracy, per variable.

    Returns (statistic, two-sided p-value). Negative statistic favours the
    model. The long-run variance uses a Newey-West/Bartlett window truncated
    at h-1, the standard choice for h-step-ahead forecast errors, which are
    MA(h-1) by construction even under the null. The Harvey, Leybourne and
    Newbold (1997) small-sample correction is applied and the statistic is
    referred to t(n-1) rather than the standard normal -- with ~49 origins the
    asymptotic normal over-rejects noticeably.

    p-values are NOT corrected for testing 8 variables x 8 horizons; callers
    comparing the whole grid should apply their own multiplicity adjustment.
    """
    d = np.square(model_err) - np.square(rw_err)  # n x k
    n = d.shape[0]
    dbar = d.mean(axis=0)
    dev = d - dbar

    gamma0 = np.mean(dev * dev, axis=0)
    lrv = gamma0.copy()
    for lag in range(1, horizon):
        if lag >= n:
            break
        cov = np.mean(dev[lag:] * dev[:-lag], axis=0)
        lrv += 2.0 * (1.0 - lag / horizon) * cov
    # A negative Bartlett estimate is possible in small samples; fall back to
    # the zero-lag variance rather than emitting a NaN statistic.
    lrv = np.where(lrv <= 0, gamma0, lrv)

    with np.errstate(divide="ignore", invalid="ignore"):
        stat = dbar / np.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold correction.
    adj = np.sqrt(max((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n, 1e-12))
    stat = stat * adj
    stat = np.where(np.isfinite(stat), stat, np.nan)

    from math import exp, lgamma, log

    def _t_sf(t: float, df: int) -> float:
        """Two-sided t tail probability via the regularized incomplete beta."""
        if not np.isfinite(t):
            return float("nan")
        x = df / (df + t * t)
        a, b = df / 2.0, 0.5
        # Continued-fraction incomplete beta (Lentz), adequate for our df.
        def _betacf(a, b, x, itmax=200, eps=3e-14):
            qab, qap, qam = a + b, a + 1.0, a - 1.0
            c, d = 1.0, 1.0 - qab * x / qap
            if abs(d) < 1e-30:
                d = 1e-30
            d = 1.0 / d
            h = d
            for m in range(1, itmax + 1):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d
                if abs(d) < 1e-30:
                    d = 1e-30
                c = 1.0 + aa / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d
                if abs(d) < 1e-30:
                    d = 1e-30
                c = 1.0 + aa / c
                if abs(c) < 1e-30:
                    c = 1e-30
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < eps:
                    break
            return h

        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        if x < (a + 1.0) / (a + b + 2.0):
            ib = exp_safe(a * log(x) + b * log(1.0 - x) - lbeta) * _betacf(a, b, x) / a
        else:
            ib = 1.0 - exp_safe(
                b * log(1.0 - x) + a * log(x) - lbeta
            ) * _betacf(b, a, 1.0 - x) / b
        return float(min(max(ib, 0.0), 1.0))

    def exp_safe(v):
        return exp(v) if v > -700 else 0.0

    df = max(n - 1, 1)
    pval = np.array([_t_sf(float(s), df) for s in stat], dtype=float)
    return stat, pval


def _drift_forecast(train: np.ndarray, horizons: int) -> np.ndarray:
    """Random walk WITH DRIFT: y_T + h * mean historical change.

    For a trending series (a log price level, log real GDP) the driftless
    random walk forfeits the whole trend as forecast error at long horizons,
    so beating it is close to uninformative. This is the textbook naive for
    an I(1)-with-drift series and is the harder, fairer benchmark.
    """
    n = train.shape[0]
    drift = (train[-1] - train[0]) / max(n - 1, 1)
    steps = np.arange(1, horizons + 1)[:, None]
    return train[-1][None, :] + drift[None, :] * steps


def _ar1_forecast(train: np.ndarray, horizons: int) -> np.ndarray:
    """Per-variable AR(1) in first differences, cumulated back to levels.

    Fitted equation by equation with an intercept; the intercept means this
    nests the drift benchmark, so it is a strictly stronger naive. Falls back
    to the drift path for any series with too little history or a degenerate
    regressor.
    """
    d = np.diff(train, axis=0)
    if d.shape[0] < 3:
        return _drift_forecast(train, horizons)
    out = np.empty((horizons, train.shape[1]))
    for j in range(train.shape[1]):
        dj = d[:, j]
        mu = float(dj.mean())
        # Mean-deviation form: (dy_t - mu) = phi (dy_{t-1} - mu). Fitting an
        # intercept instead makes the long-run per-step increment c/(1-phi),
        # which explodes as phi approaches 1 -- on this data that produced a
        # single -203 log-point 8-step path and a benchmark the model "beat"
        # 4:1 purely because the benchmark had detonated. Here the long-run
        # increment is mu by construction, so AR(1) nests the drift benchmark
        # and cannot diverge from it.
        xc, yc = dj[:-1] - mu, dj[1:] - mu
        denom = float(xc @ xc)
        phi = float(np.clip(xc @ yc / denom, -0.95, 0.95)) if denom > 0 else 0.0
        dev, level = float(dj[-1]) - mu, float(train[-1, j])
        for h in range(horizons):
            dev *= phi
            level += mu + dev
            out[h, j] = level
    return out


def rolling_origin_evaluation(
    y: np.ndarray,
    *,
    lags: int = 4,
    horizons: int = 4,
    first_origin: int | None = None,
    dummies: np.ndarray | None = None,
    bvar_kwargs: dict | None = None,
) -> dict:
    """Evaluate expanding-window posterior-mean forecasts against no change.

    The observation at each origin is included in estimation and forecasts
    start one row later. Structural identification and FEVD targets are not
    used, so the evaluation cannot select a specification for matching the
    paper's structural results.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not np.isfinite(y).all():
        raise ValueError("y must be a finite T x k array")
    T, k = y.shape
    if horizons < 1:
        raise ValueError("horizons must be positive")
    first_origin = max(40, T // 2) if first_origin is None else int(first_origin)
    last_origin = T - horizons - 1
    if first_origin <= lags + 1 or first_origin > last_origin:
        raise ValueError("evaluation window leaves no valid rolling origins")

    if dummies is not None:
        dummies = np.asarray(dummies, dtype=float)
        if dummies.ndim == 1:
            dummies = dummies[:, None]
        if dummies.shape[0] != T:
            raise ValueError("dummies must have the same rows as y")

    kwargs = dict(bvar_kwargs or {})
    model_errors = [[] for _ in range(horizons)]
    rw_errors = [[] for _ in range(horizons)]
    drift_errors = [[] for _ in range(horizons)]
    ar1_errors = [[] for _ in range(horizons)]
    actuals = [[] for _ in range(horizons)]
    points = [[] for _ in range(horizons)]
    origins = list(range(first_origin, last_origin + 1))
    for origin in origins:
        train = y[: origin + 1]
        train_dummies = None if dummies is None else dummies[: origin + 1]
        model = BVAR(train, lags=lags, dummies=train_dummies, **kwargs)
        sigma_mean = model.S_post / max(model.df_post - k - 1, 1)
        draw = PosteriorDraw(
            Pi=model.B_post.T.copy(), Sigma=sigma_mean, lags=lags, k=k
        )
        forecast = unconditional_forecast(draw, train, horizons=horizons)
        actual = y[origin + 1 : origin + horizons + 1]
        random_walk = np.repeat(train[-1][None, :], horizons, axis=0)
        drift = _drift_forecast(train, horizons)
        ar1 = _ar1_forecast(train, horizons)
        for h in range(horizons):
            model_errors[h].append(forecast[h] - actual[h])
            rw_errors[h].append(random_walk[h] - actual[h])
            drift_errors[h].append(drift[h] - actual[h])
            ar1_errors[h].append(ar1[h] - actual[h])
            actuals[h].append(actual[h])
            points[h].append(forecast[h])

    rows = []
    for h in range(horizons):
        model_rmse = _rmse(np.asarray(model_errors[h]))
        rw_rmse = _rmse(np.asarray(rw_errors[h]))
        relative = np.divide(
            model_rmse,
            rw_rmse,
            out=np.full_like(model_rmse, np.nan),
            where=rw_rmse > 0,
        )
        me = np.asarray(model_errors[h])
        re_ = np.asarray(rw_errors[h])
        de = np.asarray(drift_errors[h])
        ae = np.asarray(ar1_errors[h])
        dm_stat, dm_p = _diebold_mariano(me, re_, h + 1)
        dm_stat_drift, dm_p_drift = _diebold_mariano(me, de, h + 1)
        dm_stat_ar1, dm_p_ar1 = _diebold_mariano(me, ae, h + 1)
        drift_rmse = _rmse(de)
        ar1_rmse = _rmse(ae)

        def _mse_share(err):
            """Fraction of a benchmark's MSE contributed by its single worst
            origin. An AR(1) extrapolating the 2020Q2 collapse puts ~95% of
            its 8-step MSE in one origin, which makes the resulting ratio
            arithmetically correct and evidentially worthless. Emitted so no
            consumer can plot such a ratio without seeing that."""
            sq = np.square(err)
            tot = sq.sum(axis=0)
            return np.divide(
                sq.max(axis=0), tot,
                out=np.full(err.shape[1], np.nan), where=tot > 0,
            )

        def _ratio(bench):
            return np.divide(
                model_rmse, bench,
                out=np.full_like(model_rmse, np.nan), where=bench > 0,
            )
        rows.append({
            "horizon": h + 1,
            "bvar_rmse": model_rmse.tolist(),
            "random_walk_rmse": rw_rmse.tolist(),
            "relative_rmse": relative.tolist(),
            "variables_beating_random_walk": int(np.sum(relative < 1)),
            # Significance of the accuracy difference. Without this a ratio of
            # 0.97 reads as a win and 1.06 as a loss when neither may be
            # distinguishable from the benchmark.
            "dm_stat": [None if not np.isfinite(v) else float(v) for v in dm_stat],
            "dm_pvalue": [None if not np.isfinite(v) else float(v) for v in dm_p],
            "n_origins": int(me.shape[0]),
            # Per-origin errors, so downstream work (bootstrap bands, error
            # decomposition by episode, alternative losses) does not require
            # re-running the whole evaluation.
            "bvar_errors_by_origin": me.tolist(),
            "random_walk_errors_by_origin": re_.tolist(),
            # Harder naive benchmarks. A driftless random walk on a trending
            # log level forfeits the trend by construction, so a low ratio
            # against it is close to uninformative for price and output
            # levels; the drift and AR(1) paths are the fair comparison.
            "drift_rmse": drift_rmse.tolist(),
            "ar1_rmse": ar1_rmse.tolist(),
            "relative_rmse_vs_drift": _ratio(drift_rmse).tolist(),
            "relative_rmse_vs_ar1": _ratio(ar1_rmse).tolist(),
            "dm_stat_vs_drift": [None if not np.isfinite(v) else float(v) for v in dm_stat_drift],
            "dm_pvalue_vs_drift": [None if not np.isfinite(v) else float(v) for v in dm_p_drift],
            "dm_stat_vs_ar1": [None if not np.isfinite(v) else float(v) for v in dm_stat_ar1],
            "dm_pvalue_vs_ar1": [None if not np.isfinite(v) else float(v) for v in dm_p_ar1],
            "worst_origin_mse_share": {
                "bvar": _mse_share(me).tolist(),
                "random_walk": _mse_share(re_).tolist(),
                "drift": _mse_share(de).tolist(),
                "ar1": _mse_share(ae).tolist(),
            },
            "drift_errors_by_origin": de.tolist(),
            "ar1_errors_by_origin": ae.tolist(),
            # Actuals and point forecasts, so Mincer-Zarnowitz regressions,
            # MASE and episode splits are computable from this file alone.
            "actuals_by_origin": np.asarray(actuals[h]).tolist(),
            "bvar_point_by_origin": np.asarray(points[h]).tolist(),
        })

    return {
        "method": "expanding-window pseudo-out-of-sample evaluation",
        "origin_index": origins,
        "lags": lags,
        "origins": len(origins),
        "first_origin": first_origin,
        "last_origin": last_origin,
        "horizons": rows,
        "limitations": [
            "Errors use the transformed model units supplied by the caller.",
            "This evaluates reduced-form forecasts, not structural identification.",
            "No-change is a minimum benchmark, not an official forecast comparison.",
            "Diebold-Mariano p-values are per (variable, horizon) and are not "
            "adjusted for testing the whole grid.",
            "Estimation uses final (revised) data, so this is pseudo- not "
            "real-time out-of-sample.",
        ],
    }

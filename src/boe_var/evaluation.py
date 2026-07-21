"""Leakage-safe pseudo-out-of-sample forecast evaluation."""

from __future__ import annotations

import numpy as np

from .bvar import BVAR, PosteriorDraw
from .forecast import unconditional_forecast

__all__ = ["rolling_origin_evaluation"]


def _rmse(errors: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(errors), axis=0))


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
        for h in range(horizons):
            model_errors[h].append(forecast[h] - actual[h])
            rw_errors[h].append(random_walk[h] - actual[h])

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
        rows.append({
            "horizon": h + 1,
            "bvar_rmse": model_rmse.tolist(),
            "random_walk_rmse": rw_rmse.tolist(),
            "relative_rmse": relative.tolist(),
            "variables_beating_random_walk": int(np.sum(relative < 1)),
        })

    return {
        "method": "expanding-window pseudo-out-of-sample evaluation",
        "lags": lags,
        "origins": len(origins),
        "first_origin": first_origin,
        "last_origin": last_origin,
        "horizons": rows,
        "limitations": [
            "Errors use the transformed model units supplied by the caller.",
            "This evaluates reduced-form forecasts, not structural identification.",
            "No-change is a minimum benchmark, not an official forecast comparison.",
        ],
    }

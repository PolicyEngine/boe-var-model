"""Covid volatility scaling for the BVAR (Lenza & Primiceri, 2022).

Lenza, M. and Primiceri, G. E. (2022), "How to estimate a vector
autoregression after March 2020", Journal of Applied Econometrics 37(4),
688-699. Instead of dummying the Covid quarters out, the innovation variance
is scaled up in those quarters,

    y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + c + s_t u_t,   u_t ~ N(0, Sigma),

with s_t = 1 before the pandemic, three free scales s_0, s_1, s_2 for the
first three Covid periods and a geometric decay back to one afterwards,

    s_{t*+j} = 1 + (s_2 - 1) rho^(j-2),   j >= 3.

Here the data are quarterly and t* = 2020Q2 (the first quarter with the
lockdown collapse in it), so s_0, s_1, s_2 scale 2020Q2, 2020Q3 and 2020Q4.
Conditional on the scale path the model is a weighted (GLS) BVAR: dividing
row t of Y and X by s_t gives back the conjugate NIW posterior, and the
marginal likelihood picks up the Jacobian -k sum_t log s_t. The scale
parameters are set at their posterior mode -- log marginal likelihood plus
LP's hyperpriors (Pareto(1, 1) on each s, Beta with mode 0.8 and standard
deviation 0.2 on rho) -- with the VAR coefficients and Sigma integrated out
analytically, which is LP's empirical-Bayes step. The Minnesota tightness
(lam, mu, theta) stays at the repo defaults; it is not re-optimised.

Forecasts scale the period-(T+h) innovation by s_{T+h} = 1 + (s_T - 1) rho^h,
which is LP's law of motion from s_T on. That is what lets the predictive
bands widen right after the shock and narrow back as the volatility decays.

Opt-in only: nothing in the published specification calls this module, and
with every s_t = 1 ``ScaledBVAR`` and ``sample_forecast_scaled`` reproduce
``BVAR`` and ``forecast.sample_forecast`` bit for bit.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import brentq, minimize
from scipy.special import betaln

from . import forecast as fc
from .bvar import BVAR

__all__ = [
    "COVID_START",
    "RHO_PRIOR",
    "ScaledBVAR",
    "beta_from_mode_sd",
    "estimate_covid_scale",
    "future_scale",
    "sample_forecast_scaled",
    "scale_path",
]

COVID_START = "2020Q2"
N_FREE_SCALES = 3


def beta_from_mode_sd(mode: float, sd: float) -> tuple[float, float]:
    """Beta(a, b) with the given mode and standard deviation (a, b > 1)."""
    def sd_of(n):
        a = mode * (n - 2) + 1
        b = n - a
        return np.sqrt(a * b / (n * n * (n + 1))) - sd

    n = brentq(sd_of, 2.0 + 1e-9, 1e6)
    a = mode * (n - 2) + 1
    return float(a), float(n - a)


RHO_MODE, RHO_SD = 0.8, 0.2
RHO_PRIOR = beta_from_mode_sd(RHO_MODE, RHO_SD)


def _covid_offsets(index, start: str = COVID_START) -> np.ndarray:
    """Quarters since ``start`` for each row (negative before it)."""
    import pandas as pd

    t0 = pd.Period(start, "Q")
    return np.array([(q - t0).n for q in pd.PeriodIndex(index, freq="Q")])


def scale_path(index, params: dict, start: str = COVID_START) -> np.ndarray:
    """s_t for every row of ``index`` (a quarterly PeriodIndex).

    ``params`` holds ``s0``, ``s1``, ``s2`` and ``rho``. Rows before
    ``start`` get exactly 1.0.
    """
    j = _covid_offsets(index, start)
    s = np.ones(j.size)
    free = [params["s0"], params["s1"], params["s2"]]
    for n in range(N_FREE_SCALES):
        s[j == n] = free[n]
    later = j >= N_FREE_SCALES
    s[later] = 1.0 + (params["s2"] - 1.0) * params["rho"] ** (
        j[later] - (N_FREE_SCALES - 1))
    return s


def future_scale(index, params: dict, horizons: int,
                 start: str = COVID_START) -> np.ndarray:
    """Innovation scales for T+1..T+H given data through the last row.

    s_{T+h} = 1 + (s_T - 1) rho^h from the in-sample s_T, which equals LP's
    path from 2020Q4 on. Before ``start`` is in the sample every scale is 1.
    """
    s_T = scale_path(index, params, start)[-1]
    h = np.arange(1, horizons + 1)
    return 1.0 + (s_T - 1.0) * params["rho"] ** h


class ScaledBVAR(BVAR):
    """``BVAR`` with a known innovation-scale path s_t (row t of Y, X / s_t).

    ``scale``: (T,) positive array aligned with the rows of ``y`` (only rows
    ``lags..T-1`` enter), or None for the plain model. The prior dummies are
    not scaled. ``Y``, ``X`` and ``residuals()`` stay in data units, so
    ``residuals(draw) / s_t`` are the standardised innovations.
    """

    def __init__(self, y, lags: int = 4, dummies=None, scale=None, **kwargs):
        super().__init__(y, lags=lags, dummies=dummies, **kwargs)
        self.scale = None
        self._log_scale_sum = 0.0
        if scale is None:
            return
        scale = np.asarray(scale, dtype=float)
        if scale.shape != (self.y.shape[0],):
            raise ValueError("scale must have one entry per row of y")
        if not (np.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError("scale must be finite and positive")
        self.scale = scale
        w = scale[self.lags:]
        n = self.Y.shape[0]
        Yd, Xd = self._Ys[n:], self._Xs[n:]
        self._Ys = np.vstack([self.Y / w[:, None], Yd])
        self._Xs = np.vstack([self.X / w[:, None], Xd])
        # same steps, in the same order, as BVAR.__init__
        XtX = self._Xs.T @ self._Xs
        XtY = self._Xs.T @ self._Ys
        C = cholesky(XtX, lower=True)
        self._chol_XtX = C
        Bhat = solve_triangular(C.T, solve_triangular(C, XtY, lower=True),
                                lower=False)
        self.B_post = Bhat
        E = self._Ys - self._Xs @ Bhat
        self.S_post = E.T @ E
        self._log_scale_sum = float(np.log(w).sum())

    def log_marginal_likelihood(self) -> float:
        """Parent value on the weighted sample minus the Jacobian
        k * sum_t log s_t (the density of y_t = ... + s_t u_t)."""
        return super().log_marginal_likelihood() - self.k * self._log_scale_sum


def _log_prior(s, rho) -> float:
    """Pareto(1, 1) on each scale (density s^-2 on s >= 1), Beta on rho."""
    a, b = RHO_PRIOR
    lp = -2.0 * float(np.sum(np.log(s)))
    if rho is not None:
        lp += (a - 1) * np.log(rho) + (b - 1) * np.log1p(-rho) - betaln(a, b)
    return lp


def estimate_covid_scale(y, index, *, lags: int = 4, dummies=None,
                         start: str = COVID_START, bvar_kwargs: dict | None
                         = None) -> dict:
    """Posterior-mode (s0, s1, s2, rho) on the sample ``y`` / ``index``.

    Only the parameters the sample identifies are estimated: s_n once the
    n-th Covid quarter is in the sample, rho once the first quarter after
    the three free ones is. The rest stay at their prior modes (s = 1,
    rho = 0.8) and are listed in ``fixed``. A sample ending before ``start``
    returns s = 1 throughout, so the model is exactly the plain BVAR.
    """
    kw = dict(bvar_kwargs or {})
    j = _covid_offsets(index, start)
    n_cov = int(np.sum(j >= 0))
    n_s = min(n_cov, N_FREE_SCALES)
    has_rho = n_cov > N_FREE_SCALES
    base = {"s0": 1.0, "s1": 1.0, "s2": 1.0, "rho": RHO_MODE}
    names = [f"s{n}" for n in range(n_s)] + (["rho"] if has_rho else [])
    fixed = [nm for nm in ("s0", "s1", "s2", "rho") if nm not in names]
    if not names:
        m = ScaledBVAR(y, lags=lags, dummies=dummies, **kw)
        return {**base, "log_ml": m.log_marginal_likelihood(),
                "log_post": m.log_marginal_likelihood(), "estimated": [],
                "fixed": fixed, "n_covid_quarters": n_cov}

    def unpack(x):
        p = dict(base)
        for nm, v in zip(names, x):
            p[nm] = float(v)
        return p

    def neg_log_post(x):
        p = unpack(x)
        try:
            m = ScaledBVAR(y, lags=lags, dummies=dummies,
                           scale=scale_path(index, p, start), **kw)
            ll = m.log_marginal_likelihood()
        except (ValueError, np.linalg.LinAlgError):
            return 1e12
        s = [p[nm] for nm in names if nm != "rho"]
        return -(ll + _log_prior(s, p["rho"] if has_rho else None))

    bounds = [(1.0, 100.0)] * n_s + ([(0.01, 0.99)] if has_rho else [])
    best = None
    for s_init in (2.0, 5.0, 15.0):
        x0 = [s_init] * n_s + ([RHO_MODE] if has_rho else [])
        res = minimize(neg_log_post, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    p = unpack(best.x)
    m = ScaledBVAR(y, lags=lags, dummies=dummies,
                   scale=scale_path(index, p, start), **kw)
    return {**p, "log_ml": float(m.log_marginal_likelihood()),
            "log_post": float(-best.fun), "estimated": names, "fixed": fixed,
            "n_covid_quarters": n_cov}


def sample_forecast_scaled(draw, y_hist, horizons: int = 13, rng=None,
                           shock_scale=None):
    """``forecast.sample_forecast`` with the step-h innovation times
    ``shock_scale[h]``. Draws the normals exactly as ``sample_forecast``
    does, so ``shock_scale`` of ones (or None) gives the identical path."""
    if rng is None:
        rng = np.random.default_rng()
    Sigma = np.asarray(draw.Sigma, dtype=float)
    k = draw.k
    w, V = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    Ls = V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    shocks = rng.standard_normal((horizons, k)) @ Ls.T
    if shock_scale is not None:
        shocks = shocks * np.asarray(shock_scale, dtype=float)[:, None]
    return fc._iterate(draw, y_hist, horizons, shocks=shocks)

"""Bayesian VAR with conjugate Normal-Inverse-Wishart (Minnesota/GLP) prior.

Implemented via dummy-observation augmentation so the posterior is a
standard NIW (Giannone, Lenza & Primiceri, 2015 style).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import invwishart

__all__ = ["BVAR", "PosteriorDraw"]


@dataclass
class PosteriorDraw:
    """One posterior draw.

    Attributes
    ----------
    Pi : np.ndarray, shape (k, m) with m = k*lags + 1 + n_dummies
        Coefficient matrix, one row per equation. Column order (must match
        the regressor layout documented in :class:`BVAR`):
        [A_1 | A_2 | ... | A_p | c | D], i.e. columns 0..k-1 are the
        coefficients on y_{t-1}, columns k..2k-1 on y_{t-2}, ..., then the
        constant, then the exogenous 0/1 dummy columns.
    Sigma : np.ndarray, shape (k, k)
        Reduced-form innovation covariance.
    lags : int
    k : int
    """

    Pi: np.ndarray
    Sigma: np.ndarray
    lags: int
    k: int

    def companion(self) -> np.ndarray:
        """Companion matrix F (k*p x k*p) built from the lag blocks of Pi.

        F = [[A_1 ... A_p], [I, 0]] so IRFs at horizon h come from F**h.
        """
        k, p = self.k, self.lags
        F = np.zeros((k * p, k * p))
        F[:k, :] = self.Pi[:, : k * p]
        if p > 1:
            F[k:, : k * (p - 1)] = np.eye(k * (p - 1))
        return F


class BVAR:
    """Conjugate NIW Bayesian VAR with Minnesota + sum-of-coefficients prior.

    Model (per effective observation t = lags..T-1):

        y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + c + D d_t + u_t,
        u_t ~ N(0, Sigma)

    Regressor layout (order is a contract other modules rely on)
    -------------------------------------------------------------
    Each row of the regressor matrix X (shape T_eff x m, m = k*p + 1 + n_d) is

        x_t = [y_{t-1}', y_{t-2}', ..., y_{t-p}', 1, d_t']

    i.e. lag blocks first (lag 1 through lag p, each of length k, variables
    in the order of the columns of ``y``), then the constant, then the
    exogenous dummy columns in the order supplied. ``Pi`` in each
    :class:`PosteriorDraw` is k x m with the same column order, so fitted
    values are ``X @ Pi.T``.

    Prior (via dummy observations)
    ------------------------------
    - Minnesota: random-walk prior mean (unit own first lag, ``delta``),
      overall tightness ``lam`` (default 0.2), lag decay 1/l**2,
      per-equation scales s_i from AR(1) residual std deviations.
    - Sum-of-coefficients (Doan-Litterman-Sims) with tightness ``mu``
      (default 1.0), centered on pre-sample means.
    - Inverse-Wishart scale dummies diag(s).
    - Constant and dummies: loose (near-flat) prior, precision ``eps_const``.

    Parameters
    ----------
    y : (T, k) array of (transformed) data.
    lags : number of lags p.
    dummies : optional (T, n_d) array of exogenous 0/1 columns (e.g. Covid
        quarter dummies), aligned with the rows of ``y``.
    lam : overall Minnesota tightness (smaller = tighter).
    mu : sum-of-coefficients tightness (smaller = tighter).
    delta : prior mean on own first lag; scalar or length-k array.
        Use 1.0 for (log-)level variables (random walk).
    eps_const : dummy-observation weight on constant/dummy columns; small
        values make the prior on them loose.
    """

    def __init__(
        self,
        y: np.ndarray,
        lags: int = 4,
        dummies: np.ndarray | None = None,
        lam: float = 0.2,
        mu: float = 1.0,
        delta: float | np.ndarray = 1.0,
        eps_const: float = 1e-3,
    ):
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError("y must be a T x k array")
        T, k = y.shape
        if T <= lags + 1:
            raise ValueError("not enough observations for given lags")
        self.y = y
        self.k = k
        self.lags = int(lags)
        self.lam = float(lam)
        self.mu = float(mu)
        self.delta = np.full(k, delta, dtype=float) if np.isscalar(delta) else np.asarray(delta, float)
        self.eps_const = float(eps_const)

        if dummies is not None:
            dummies = np.asarray(dummies, dtype=float)
            if dummies.ndim == 1:
                dummies = dummies[:, None]
            if dummies.shape[0] != T:
                raise ValueError("dummies must have same number of rows as y")
        self.dummies = dummies
        self.n_dummies = 0 if dummies is None else dummies.shape[1]
        self.m = k * self.lags + 1 + self.n_dummies

        # --- data matrices ---------------------------------------------
        p = self.lags
        self.Y = y[p:]                                   # (T-p, k)
        Xlag = np.hstack([y[p - l : T - l] for l in range(1, p + 1)])
        cols = [Xlag, np.ones((T - p, 1))]
        if dummies is not None:
            cols.append(dummies[p:])
        self.X = np.hstack(cols)                          # (T-p, m)

        # --- per-equation AR(1) residual scales -------------------------
        s = np.empty(k)
        for i in range(k):
            yi = y[:, i]
            Z = np.column_stack([np.ones(T - 1), yi[:-1]])
            b, *_ = np.linalg.lstsq(Z, yi[1:], rcond=None)
            r = yi[1:] - Z @ b
            s[i] = max(np.sqrt(r @ r / max(T - 3, 1)), 1e-8)
        self.scales = s

        # --- dummy observations ------------------------------------------
        Yd, Xd = self._prior_dummies(s)
        self._Ys = np.vstack([self.Y, Yd])
        self._Xs = np.vstack([self.X, Xd])

        # --- posterior moments (stable via Cholesky) ----------------------
        XtX = self._Xs.T @ self._Xs
        XtY = self._Xs.T @ self._Ys
        C = cholesky(XtX, lower=True)                     # C C' = XtX
        self._chol_XtX = C
        # B_hat = XtX^{-1} XtY via two triangular solves
        Bhat = solve_triangular(C.T, solve_triangular(C, XtY, lower=True), lower=False)
        self.B_post = Bhat                                # (m, k), Pi = B.T
        E = self._Ys - self._Xs @ Bhat
        self.S_post = E.T @ E                             # IW scale
        self.df_post = self._Ys.shape[0] - self.m         # IW dof
        if self.df_post <= k + 1:
            raise ValueError("posterior degrees of freedom too small")

    # ------------------------------------------------------------------
    def _prior_dummies(self, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k, p, m = self.k, self.lags, self.m
        lam, mu = self.lam, self.mu
        rows_Y, rows_X = [], []

        # Minnesota tightness on lag coefficients: prior std of coeff on
        # var j lag l in eq i is lam * s_i / (l^2 * s_j)  -> dummy rows
        for l in range(1, p + 1):
            Yb = np.zeros((k, k))
            if l == 1:
                Yb = np.diag(self.delta * s) / lam
            Xb = np.zeros((k, m))
            Xb[:, (l - 1) * k : l * k] = np.diag(s) * (l ** 2) / lam
            rows_Y.append(Yb)
            rows_X.append(Xb)

        # IW scale dummies: pin Sigma prior scale to diag(s^2)
        rows_Y.append(np.diag(s))
        rows_X.append(np.zeros((k, m)))

        # Loose prior on constant + exogenous dummies
        n_det = 1 + self.n_dummies
        Yb = np.zeros((n_det, k))
        Xb = np.zeros((n_det, m))
        Xb[:, k * p :] = np.eye(n_det) * self.eps_const
        rows_Y.append(Yb)
        rows_X.append(Xb)

        # Sum-of-coefficients dummies (no-cointegration prior):
        # centered on pre-sample means ybar, tightness mu
        ybar = self.y[: p].mean(axis=0)
        Yb = np.diag(self.delta * ybar) / mu
        Xb = np.zeros((k, m))
        for l in range(p):
            Xb[:, l * k : (l + 1) * k] = np.diag(self.delta * ybar) / mu
        rows_Y.append(Yb)
        rows_X.append(Xb)

        return np.vstack(rows_Y), np.vstack(rows_X)

    # ------------------------------------------------------------------
    def sample_posterior(self, n_draws: int, seed: int | None = None) -> list[PosteriorDraw]:
        """Draw (Pi, Sigma) from the NIW posterior.

        Sigma ~ IW(S_post, df_post); Pi | Sigma ~ matric normal with mean
        B_post', row cov Sigma, column cov (X*'X*)^{-1}.
        """
        rng = np.random.default_rng(seed)
        k, m = self.k, self.m
        C = self._chol_XtX  # lower, C C' = XtX
        draws = []
        for _ in range(n_draws):
            Sigma = invwishart.rvs(df=self.df_post, scale=self.S_post, random_state=rng)
            Sigma = np.atleast_2d(Sigma)
            Ls = cholesky(Sigma, lower=True)
            Z = rng.standard_normal((m, k))
            # V = C'^{-1} Z has cov (XtX)^{-1} per column
            V = solve_triangular(C.T, Z, lower=False)
            B = self.B_post + V @ Ls.T                    # (m, k)
            draws.append(PosteriorDraw(Pi=B.T.copy(), Sigma=Sigma, lags=self.lags, k=k))
        return draws

    # ------------------------------------------------------------------
    def residuals(self, draw: PosteriorDraw) -> np.ndarray:
        """Reduced-form residuals u_t = y_t - Pi x_t for the actual sample.

        Returns a (T - lags, k) array aligned with rows lags..T-1 of ``y``.
        """
        return self.Y - self.X @ draw.Pi.T

"""Variable-specific stochastic volatility for the BVAR (Cogley-Sargent / Clark).

Cogley, T. and Sargent, T. J. (2005), "Drifts and volatilities: monetary
policies and outcomes in the post WWII US", Review of Economic Dynamics 8,
262-302; Clark, T. E. (2011), "Real-time density forecasts from Bayesian
vector autoregressions with stochastic volatility", JBES 29(3), 327-341.
Each reduced-form shock has its own log-volatility random walk:

    y_t = Pi x_t + u_t,     u_t ~ N(0, Sigma_t),
    Sigma_t = A^{-1} D_t A^{-1}',   D_t = diag(d_i exp(h_it)),
    h_it = h_i,t-1 + nu_it,          nu_it ~ N(0, sigma2_i),

with A unit lower triangular (variables in data order). As in
``common_sv`` the pre-sample log-volatilities are pinned, h_i0 = 0, so
Sigma = A^{-1} diag(d) A^{-1}' is the covariance of the quarter before the
first effective observation and exp(h_it) is variance relative to it. The
Minnesota / sum-of-coefficients / single-unit-root dummy rows of ``BVAR``
are kept as observations dated at that reference quarter (covariance
Sigma), exactly as ``common_sv`` keeps them unweighted, so with h = 0 the
conditional prior of Pi is the conjugate one.

Gibbs sampler, one sweep:

1. Pi | A, d, h: Gaussian GLS. With Sigma_t^{-1} = sum_l a_l a_l' / D_tl
   (a_l row l of A) the precision of vec(B) (B = Pi', column per equation)
   is sum_l kron(a_l a_l', X' W_l X), W_l = diag(1 / D_tl), dummy rows at
   weight 1 / d_l. One joint (k m)-dimensional draw: no equation-by-
   equation approximation (cf. the Carriero-Chan-Clark-Marcellino 2022
   corrigendum to the triangular algorithm).
2. (A, d) | Pi, h: equation i regresses u_it on -u_<i,t with weights
   exp(-h_it) (dummy rows weight 1). Prior p(Sigma) ~ |Sigma|^{-(k+1)/2},
   the improper prior behind ``BVAR``'s IW(S_post, T* - m) posterior; under
   the triangular change of variables d_i | Pi, h ~ IG((T* - k + i) / 2,
   SSR_i / 2) (i = 1..k, a_i integrated out) and a_i | d_i ~ N(OLS,
   d_i (Z'Z)^{-1}). With h = 0 steps 1-2 are an exact Gibbs sampler for the
   NIW posterior of ``BVAR``.
3. h_i | Pi, A, d: e_it = (A u_t)_i / sqrt(d_i) ~ N(0, exp(h_it)); KSC
   seven-component mixture for log e^2 and forward-filtering backward-
   sampling, all k scalar states at once.
4. sigma2_i | h_i: inverse gamma, prior ``common_sv.SIGMA2_PRIOR``.

Identification: ``identification.identify`` uses a draw only through
``draw.Sigma``. Because Sigma_t is no longer a scalar multiple of Sigma, the
admissible rotations and the impact matrix B_t = chol(Sigma_t) Q_t change
with t; the zero restrictions are imposed on the impact matrix at whatever
date is identified, so they hold at every date by construction, but the
structural shocks' relative sizes (and, through Q_t, their identification)
drift. ``VSVDraw.at(h)`` returns the draw with Sigma replaced by Sigma at
log-volatility vector h; pass ``draw.at(draw.h_T)`` to identify at the
forecast origin. ``draw.Sigma`` itself is the reference-quarter covariance.

Opt-in only: nothing in the published specification calls this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from . import forecast as fc
from .bvar import BVAR, PosteriorDraw
from .common_sv import KSC_MEANS, KSC_PROBS, KSC_VARS, OFFSET, SIGMA2_PRIOR

__all__ = ["VariableSVBVAR", "VSVDraw", "sample_forecast_vsv",
           "sigma_from_factors"]


def sigma_from_factors(A, d):
    """A^{-1} diag(d) A^{-1}'."""
    Ai = solve_triangular(np.asarray(A, float), np.eye(len(d)), lower=True,
                          unit_diagonal=True)
    return (Ai * np.asarray(d, float)) @ Ai.T


@dataclass
class VSVDraw(PosteriorDraw):
    """``PosteriorDraw`` plus the triangular factors and volatility state.

    ``Sigma`` = A^{-1} diag(d) A^{-1}' is the reference-quarter (h = 0)
    covariance; ``h_T`` (k,) the last-sample log-volatilities; ``sigma2``
    (k,) the random-walk innovation variances.
    """

    A: np.ndarray = field(default=None)
    d: np.ndarray = field(default=None)
    h_T: np.ndarray = field(default=None)
    sigma2: np.ndarray = field(default=None)

    def at(self, h) -> "VSVDraw":
        """The same draw with Sigma at log-volatility vector ``h``."""
        return replace(self, Sigma=sigma_from_factors(
            self.A, self.d * np.exp(np.asarray(h, float))))


class VariableSVBVAR:
    """BVAR with one random-walk log-volatility per reduced-form shock.

    ``y``, ``lags``, ``dummies`` and ``**bvar_kwargs`` are those of
    :class:`~boe_var.bvar.BVAR`; its prior dummy rows are reused unchanged.
    """

    def __init__(self, y, lags: int = 4, dummies=None,
                 sigma2_prior=SIGMA2_PRIOR, offset: float = OFFSET,
                 **bvar_kwargs):
        self.base = b = BVAR(y, lags=lags, dummies=dummies, **bvar_kwargs)
        n = b.Y.shape[0]
        self.Y, self.X = b.Y, b.X
        self._Yd, self._Xd = b._Ys[n:], b._Xs[n:]
        self.k, self.m, self.lags, self.n_obs = b.k, b.m, b.lags, n
        self.n_rows = b._Ys.shape[0]                       # T*
        self.sigma2_prior = tuple(float(v) for v in sigma2_prior)
        self.offset = float(offset)
        self._XdXd = self._Xd.T @ self._Xd
        self._XdYd = self._Xd.T @ self._Yd
        self.h_mean = None          # (n, k) posterior mean after sample()
        self.h_bands = None         # (3, n, k): 5 / 50 / 95 percentiles

    # ------------------------------------------------------------------
    def _pi_moments(self, A, d, h):
        """Precision P and mean (m, k) of B = Pi' given (A, d, h)."""
        k, m = self.k, self.m
        X, Y = self.X, self.Y
        W = np.exp(-h) / d                                  # (n, k)
        P = np.zeros((k * m, k * m))
        M = np.zeros((m, k))
        for l in range(k):
            a = A[l]
            XW = X * W[:, l:l + 1]
            N = XW.T @ X + self._XdXd / d[l]
            r = XW.T @ Y + self._XdYd / d[l]
            aa = np.outer(a, a)
            P += np.kron(aa, N)
            M += r @ aa
        C = cholesky(P, lower=True)
        mean = cho_solve((C, True), M.reshape(-1, order="F"))
        return C, mean.reshape((m, k), order="F")

    def _draw_pi(self, A, d, h, rng):
        C, mean = self._pi_moments(A, d, h)
        z = rng.standard_normal(self.k * self.m)
        v = solve_triangular(C.T, z, lower=False)
        B = mean + v.reshape((self.m, self.k), order="F")
        return B.T.copy()

    def _draw_A_d(self, Pi, h, rng):
        k = self.k
        U = self.Y - self.X @ Pi.T
        Ud = self._Yd - self._Xd @ Pi.T
        A = np.eye(k)
        d = np.empty(k)
        for i in range(k):
            s = np.exp(-0.5 * h[:, i])[:, None]
            z = np.concatenate([U[:, i] * s[:, 0], Ud[:, i]])
            alpha = 0.5 * (self.n_rows - k + i + 1)
            if i == 0:
                ssr = float(z @ z)
                d[i] = 1.0 / rng.gamma(alpha, 2.0 / ssr)
                continue
            Z = -np.vstack([U[:, :i] * s, Ud[:, :i]])
            ZtZ = Z.T @ Z
            L = cholesky(ZtZ, lower=True)
            ahat = cho_solve((L, True), Z.T @ z)
            e = z - Z @ ahat
            ssr = float(e @ e)
            d[i] = 1.0 / rng.gamma(alpha, 2.0 / ssr)
            A[i, :i] = ahat + np.sqrt(d[i]) * solve_triangular(
                L.T, rng.standard_normal(i), lower=False)
        return A, d

    def _draw_h(self, Pi, A, d, sigma2, rng):
        """KSC indicators, then FFBS for all k random-walk states at once."""
        U = self.Y - self.X @ Pi.T
        E = (U @ A.T) / np.sqrt(d)                         # (n, k)
        z = np.log(E ** 2 + self.offset)
        resid = z[:, :, None] - self._h[:, :, None] - KSC_MEANS
        logp = (np.log(KSC_PROBS) - 0.5 * np.log(KSC_VARS)
                - 0.5 * resid ** 2 / KSC_VARS)
        logp -= logp.max(axis=-1, keepdims=True)
        cdf = np.cumsum(np.exp(logp), axis=-1)
        u = rng.random(z.shape) * cdf[..., -1]
        s = (u[..., None] > cdf).sum(axis=-1)
        R = KSC_VARS[s]
        obs = z - KSC_MEANS[s]
        n, k = z.shape
        mf, Pf = np.empty((n, k)), np.empty((n, k))
        a, P = np.zeros(k), sigma2.copy()                  # h_0 = 0 known
        for t in range(n):
            if t:
                a, P = mf[t - 1], Pf[t - 1] + sigma2
            K = P / (P + R[t])
            mf[t] = a + K * (obs[t] - a)
            Pf[t] = (1.0 - K) * P
        out = np.empty((n, k))
        e = rng.standard_normal((n, k))
        out[-1] = mf[-1] + np.sqrt(Pf[-1]) * e[-1]
        for t in range(n - 2, -1, -1):
            G = Pf[t] / (Pf[t] + sigma2)
            mean = mf[t] + G * (out[t + 1] - mf[t])
            var = np.maximum(Pf[t] * (1.0 - G), 0.0)
            out[t] = mean + np.sqrt(var) * e[t]
        return out

    def _draw_sigma2(self, h, rng):
        a0, b0 = self.sigma2_prior
        lag = np.vstack([np.zeros((1, self.k)), h[:-1]])
        ssr = ((h - lag) ** 2).sum(axis=0)
        return 1.0 / rng.gamma(a0 + 0.5 * h.shape[0], 1.0 / (b0 + 0.5 * ssr))

    # ------------------------------------------------------------------
    def sample(self, n_draws: int, seed: int | None = None, burn: int = 1000,
               thin: int = 10, fix_volatility: bool = False) -> list[VSVDraw]:
        """Run the Gibbs sampler; return ``n_draws`` kept draws.

        ``fix_volatility=True`` holds h = 0 (every volatility at 1) and
        sigma2 = 0 and runs steps 1-2 only: a Gibbs sampler whose target is
        the homoskedastic ``BVAR`` NIW posterior. The posterior mean and
        5/50/95 percentiles of the log-volatility paths are stored on
        ``h_mean`` / ``h_bands``.
        """
        rng = np.random.default_rng(seed)
        k, n = self.k, self.n_obs
        self._h = np.zeros((n, k))
        sigma2 = np.full(k, self.sigma2_prior[1] / (self.sigma2_prior[0] - 1))
        b = self.base
        Pi = b.B_post.T.copy()
        out, hs = [], []
        for it in range(burn + n_draws * thin):
            A, d = self._draw_A_d(Pi, self._h, rng)
            Pi = self._draw_pi(A, d, self._h, rng)
            if not fix_volatility:
                self._h = self._draw_h(Pi, A, d, sigma2, rng)
                sigma2 = self._draw_sigma2(self._h, rng)
            if it >= burn and (it - burn) % thin == thin - 1:
                out.append(VSVDraw(
                    Pi=Pi, Sigma=sigma_from_factors(A, d), lags=self.lags,
                    k=k, A=A, d=d, h_T=self._h[-1].copy(),
                    sigma2=np.zeros(k) if fix_volatility else sigma2.copy()))
                hs.append(self._h.copy())
        H = np.asarray(hs)
        self.h_mean = H.mean(axis=0)
        self.h_bands = np.percentile(H, [5.0, 50.0, 95.0], axis=0)
        return out


def sample_forecast_vsv(draw: VSVDraw, y_hist, horizons: int = 13, rng=None):
    """One stochastic path: h_{T+j} = h_{T+j-1} + nu, shock A^{-1} D^{1/2} z.

    Draws z (horizons, k) first, then nu (horizons, k).
    """
    if rng is None:
        rng = np.random.default_rng()
    k = draw.k
    z = rng.standard_normal((horizons, k))
    nu = rng.standard_normal((horizons, k)) * np.sqrt(draw.sigma2)
    h = draw.h_T + np.cumsum(nu, axis=0)
    w = z * np.sqrt(draw.d * np.exp(h))
    shocks = solve_triangular(draw.A, w.T, lower=True,
                              unit_diagonal=True).T
    return fc._iterate(draw, y_hist, horizons, shocks=shocks)

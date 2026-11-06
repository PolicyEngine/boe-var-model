"""Common stochastic volatility for the BVAR (Carriero, Clark & Marcellino, 2016).

Carriero, A., Clark, T. E. and Marcellino, M. (2016), "Common drifting
volatility in large Bayesian VARs", Journal of Business & Economic
Statistics 34(3), 375-390. One log-volatility factor scales the whole
reduced-form covariance,

    y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + c + D d_t + lambda_t^{1/2} u_t,
    u_t ~ N(0, Sigma),
    h_t = log lambda_t = phi h_{t-1} + nu_t,   nu_t ~ N(0, sigma2),

with ``phi = 1`` (random walk, ``law="rw"``) or ``phi`` sampled
(``law="ar1"``, normal prior truncated to |phi| < 1). The scale of
``lambda_t Sigma`` is only identified up to a constant, so the pre-sample
log-volatility is pinned at h_0 = 0: Sigma is the covariance of the
quarter before the first effective observation and lambda_t is volatility
relative to it.

Gibbs sampler, one sweep:

1. (Pi, Sigma | h): conditional on h the model is the weighted conjugate BVAR
   of ``covid_volatility.ScaledBVAR`` with s_t = exp(h_t / 2) -- divide row t
   of Y and X by s_t and draw from the NIW posterior, with the same
   Minnesota / sum-of-coefficients / (optional) single-unit-root dummies as
   ``BVAR``. The prior dummies are not reweighted.
2. (h | Pi, Sigma, sigma2, phi): standardise the residuals,
   e_t = chol(Sigma)^{-1} u_t ~ N(0, lambda_t I_k), so each of the k
   log(e_it^2 + c) = h_t + log chi2_1. Approximate log chi2_1 by the
   seven-component normal mixture of Kim, Shephard & Chib (1998), draw the
   mixture indicators, collapse the k measurements of period t into one
   Gaussian pseudo-observation, and draw the whole path by forward-filtering
   backward-sampling (scalar state).
3. (sigma2 | h): inverse gamma. (phi | h, sigma2): truncated normal (ar1).

Identification: the structural code uses Sigma only through chol(Sigma)
(the zero-restriction null spaces, the sign checks) and |det A0|. With
Sigma_t = lambda_t Sigma, chol(Sigma_t) = lambda_t^{1/2} chol(Sigma), so the
admissible Q and the sign pattern are unchanged and the impact matrix is
B_t = lambda_t^{1/2} B. Identifying on each draw's Sigma gives the
structural model at lambda = 1 (the h_0 = 0 quarter); responses to a
one-standard-deviation shock at date t scale by lambda_t^{1/2}.

Opt-in only: nothing in the published specification calls this module.
With ``fix_volatility=True`` the sampler skips steps 2-3, uses no extra
random numbers and reproduces ``BVAR.sample_posterior`` bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import invwishart

from . import forecast as fc
from .bvar import BVAR, PosteriorDraw

__all__ = [
    "KSC_MEANS",
    "KSC_PROBS",
    "KSC_VARS",
    "CommonSVBVAR",
    "SVDraw",
    "sample_forecast_sv",
    "simulate_log_vol",
]

# Kim, Shephard & Chib (1998, Table 4): log chi2_1 ~ sum_j q_j N(m_j, v_j).
# The table's means are listed before the -1.2704 shift (E log chi2_1).
KSC_PROBS = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566,
                      0.25750])
KSC_MEANS = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942,
                      1.79518, -1.08819]) - 1.2704
KSC_VARS = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023,
                     1.26261])

# Weakly informative defaults. sigma2 ~ IG(a, b) with mean b / (a - 1) = 0.04
# (a 0.2 quarterly sd of the log-variance innovation, CCM-sized).
SIGMA2_PRIOR = (3.0, 0.08)
PHI_PRIOR = (0.9, 0.1 ** 2)     # ar1 only: N(mean, var) truncated to |phi| < 1
OFFSET = 1e-6                   # c in log(e^2 + c)


@dataclass
class SVDraw(PosteriorDraw):
    """``PosteriorDraw`` plus the volatility state needed to forecast.

    h_T: log-volatility in the last sample period; sigma2, phi: law of motion.
    ``Sigma`` is the covariance at lambda = 1 (h = 0).
    """

    h_T: float = 0.0
    sigma2: float = 0.0
    phi: float = 1.0


class CommonSVBVAR:
    """BVAR with one common stochastic volatility factor (CCM 2016).

    ``y``, ``lags``, ``dummies`` and ``**bvar_kwargs`` (lam, mu, theta, ...)
    are those of :class:`~boe_var.bvar.BVAR`, whose prior dummies are reused
    unchanged. ``law`` is ``"rw"`` (phi = 1) or ``"ar1"``.
    """

    def __init__(self, y, lags: int = 4, dummies=None, law: str = "rw",
                 sigma2_prior=SIGMA2_PRIOR, phi_prior=PHI_PRIOR,
                 offset: float = OFFSET, **bvar_kwargs):
        if law not in ("rw", "ar1"):
            raise ValueError("law must be 'rw' or 'ar1'")
        self.base = BVAR(y, lags=lags, dummies=dummies, **bvar_kwargs)
        self.law = law
        self.sigma2_prior = tuple(float(v) for v in sigma2_prior)
        self.phi_prior = tuple(float(v) for v in phi_prior)
        self.offset = float(offset)
        b = self.base
        n = b.Y.shape[0]
        self.Y, self.X = b.Y, b.X
        self._Yd, self._Xd = b._Ys[n:], b._Xs[n:]
        self.k, self.m, self.lags = b.k, b.m, b.lags
        self.df_post = b.df_post
        self.n_obs = n
        self.h_mean = None          # posterior mean path after sample()
        self.h_bands = None         # (5, 50, 95) percentiles, shape (3, n)

    # ------------------------------------------------------------------
    def _moments(self, h):
        """(chol(X*'X*), B_post, S_post) with rows weighted by exp(-h/2).

        ``h is None`` returns the homoskedastic ``BVAR`` moments themselves.
        Same operations, in the same order, as ``ScaledBVAR``.
        """
        b = self.base
        if h is None:
            return b._chol_XtX, b.B_post, b.S_post
        w = np.exp(0.5 * h)[:, None]
        Xs = np.vstack([self.X / w, self._Xd])
        Ys = np.vstack([self.Y / w, self._Yd])
        XtX = Xs.T @ Xs
        XtY = Xs.T @ Ys
        C = cholesky(XtX, lower=True)
        Bhat = solve_triangular(C.T, solve_triangular(C, XtY, lower=True),
                                lower=False)
        E = Ys - Xs @ Bhat
        return C, Bhat, E.T @ E

    def _draw_niw(self, C, B_post, S_post, rng):
        """One (Pi, Sigma) draw; the steps of ``BVAR.sample_posterior``."""
        k, m = self.k, self.m
        Sigma = invwishart.rvs(df=self.df_post, scale=S_post, random_state=rng)
        Sigma = np.atleast_2d(Sigma)
        Ls = cholesky(Sigma, lower=True)
        Z = rng.standard_normal((m, k))
        V = solve_triangular(C.T, Z, lower=False)
        B = B_post + V @ Ls.T
        return B.T.copy(), Sigma

    # ------------------------------------------------------------------
    def _draw_h(self, Pi, Sigma, h, sigma2, phi, rng):
        """KSC mixture indicators, then FFBS for the scalar log-vol path."""
        U = self.Y - self.X @ Pi.T
        L = cholesky(Sigma, lower=True)
        E = solve_triangular(L, U.T, lower=True).T            # (n, k)
        z = np.log(E ** 2 + self.offset)
        # indicators: P(s = j) ~ q_j N(z; h + m_j, v_j)
        resid = z[:, :, None] - h[:, None, None] - KSC_MEANS
        logp = (np.log(KSC_PROBS) - 0.5 * np.log(KSC_VARS)
                - 0.5 * resid ** 2 / KSC_VARS)
        logp -= logp.max(axis=-1, keepdims=True)
        p = np.exp(logp)
        cdf = np.cumsum(p, axis=-1)
        u = rng.random(z.shape) * cdf[..., -1]
        s = (u[..., None] > cdf).sum(axis=-1)
        v = KSC_VARS[s]
        prec = (1.0 / v).sum(axis=1)
        R = 1.0 / prec
        obs = R * ((z - KSC_MEANS[s]) / v).sum(axis=1)
        # forward filter, h_0 = 0 known
        n = self.n_obs
        mf, Pf = np.empty(n), np.empty(n)
        a, P = 0.0, sigma2
        for t in range(n):
            if t:
                a, P = phi * mf[t - 1], phi * phi * Pf[t - 1] + sigma2
            K = P / (P + R[t])
            mf[t] = a + K * (obs[t] - a)
            Pf[t] = (1.0 - K) * P
        # backward sample
        out = np.empty(n)
        e = rng.standard_normal(n)
        out[-1] = mf[-1] + np.sqrt(Pf[-1]) * e[-1]
        for t in range(n - 2, -1, -1):
            Pn = phi * phi * Pf[t] + sigma2
            G = Pf[t] * phi / Pn
            mean = mf[t] + G * (out[t + 1] - phi * mf[t])
            var = max(Pf[t] - G * phi * Pf[t], 0.0)
            out[t] = mean + np.sqrt(var) * e[t]
        return out

    def _draw_sigma2(self, h, phi, rng):
        a0, b0 = self.sigma2_prior
        lag = np.concatenate([[0.0], h[:-1]])
        ssr = float(np.sum((h - phi * lag) ** 2))
        return 1.0 / rng.gamma(a0 + 0.5 * h.size, 1.0 / (b0 + 0.5 * ssr))

    def _draw_phi(self, h, sigma2, phi_old, rng):
        m0, v0 = self.phi_prior
        lag = np.concatenate([[0.0], h[:-1]])
        prec = 1.0 / v0 + lag @ lag / sigma2
        mean = (m0 / v0 + lag @ h / sigma2) / prec
        for _ in range(100):
            phi = mean + rng.standard_normal() / np.sqrt(prec)
            if abs(phi) < 1.0:
                return float(phi)
        return phi_old

    # ------------------------------------------------------------------
    def sample(self, n_draws: int, seed: int | None = None, burn: int = 1000,
               thin: int = 10, fix_volatility: bool = False) -> list[SVDraw]:
        """Run the Gibbs sampler; return ``n_draws`` kept draws.

        ``fix_volatility=True``: lambda_t = 1 throughout, no burn-in or
        thinning (the NIW draws are then independent) and no extra random
        numbers, so the (Pi, Sigma) draws equal ``BVAR.sample_posterior(
        n_draws, seed)`` exactly. Otherwise ``burn`` sweeps are discarded and
        every ``thin``-th sweep kept; the posterior mean and 5/50/95
        percentiles of the log-volatility path are stored on ``h_mean`` /
        ``h_bands``.
        """
        rng = np.random.default_rng(seed)
        k, p = self.k, self.lags
        if fix_volatility:
            C, B, S = self._moments(None)
            out = []
            for _ in range(n_draws):
                Pi, Sigma = self._draw_niw(C, B, S, rng)
                out.append(SVDraw(Pi=Pi, Sigma=Sigma, lags=p, k=k))
            self.h_mean = np.zeros(self.n_obs)
            self.h_bands = np.zeros((3, self.n_obs))
            return out
        h = np.zeros(self.n_obs)
        sigma2 = self.sigma2_prior[1] / (self.sigma2_prior[0] - 1.0)
        phi = 1.0 if self.law == "rw" else self.phi_prior[0]
        out, hs = [], []
        total = burn + n_draws * thin
        for it in range(total):
            Pi, Sigma = self._draw_niw(*self._moments(h), rng)
            h = self._draw_h(Pi, Sigma, h, sigma2, phi, rng)
            sigma2 = self._draw_sigma2(h, phi, rng)
            if self.law == "ar1":
                phi = self._draw_phi(h, sigma2, phi, rng)
            if it >= burn and (it - burn) % thin == thin - 1:
                out.append(SVDraw(Pi=Pi, Sigma=Sigma, lags=p, k=k,
                                  h_T=float(h[-1]), sigma2=float(sigma2),
                                  phi=float(phi)))
                hs.append(h.copy())
        H = np.asarray(hs)
        self.h_mean = H.mean(axis=0)
        self.h_bands = np.percentile(H, [5.0, 50.0, 95.0], axis=0)
        return out


def simulate_log_vol(draw: SVDraw, horizons: int, rng) -> np.ndarray:
    """h_{T+1..T+H} from the draw's law of motion."""
    e = rng.standard_normal(horizons) * np.sqrt(draw.sigma2)
    h = np.empty(horizons)
    prev = draw.h_T
    for j in range(horizons):
        prev = draw.phi * prev + e[j]
        h[j] = prev
    return h


def sample_forecast_sv(draw: SVDraw, y_hist, horizons: int = 13, rng=None):
    """One stochastic path with innovation scale exp(h_{T+j} / 2).

    A draw with no volatility (sigma2 = 0 and h_T = 0, as from
    ``fix_volatility=True``) uses exactly the random numbers of
    ``forecast.sample_forecast`` and returns the identical path.
    """
    if rng is None:
        rng = np.random.default_rng()
    if draw.sigma2 == 0.0 and draw.h_T == 0.0:
        return fc.sample_forecast(draw, y_hist, horizons=horizons, rng=rng)
    from .covid_volatility import sample_forecast_scaled

    h = simulate_log_vol(draw, horizons, rng)
    return sample_forecast_scaled(draw, y_hist, horizons=horizons, rng=rng,
                                  shock_scale=np.exp(0.5 * h))

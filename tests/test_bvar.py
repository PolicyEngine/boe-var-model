import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boe_var.bvar import BVAR, PosteriorDraw


def simulate_var1(T=500, seed=0, dummies=None, dummy_coefs=None):
    """Simulate a stationary VAR(1), k=2, with known coefficients."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.6, 0.15], [0.1, 0.7]])
    c = np.array([0.5, -0.3])
    L = np.array([[1.0, 0.0], [0.4, 0.8]])
    y = np.zeros((T + 100, 2))
    for t in range(1, T + 100):
        y[t] = c + A @ y[t - 1] + L @ rng.standard_normal(2)
    y = y[100:]
    if dummies is not None:
        y = y + dummies @ dummy_coefs.T
    return y, A, c, L @ L.T


def test_posterior_mean_close_to_truth():
    y, A, c, Sig = simulate_var1()
    model = BVAR(y, lags=1, lam=0.5)
    draws = model.sample_posterior(500, seed=42)
    Pi_mean = np.mean([d.Pi for d in draws], axis=0)
    A_hat = Pi_mean[:, :2]
    c_hat = Pi_mean[:, 2]
    Sig_mean = np.mean([d.Sigma for d in draws], axis=0)
    assert np.allclose(A_hat, A, atol=0.1)
    assert np.allclose(c_hat, c, atol=0.25)
    assert np.allclose(Sig_mean, Sig, atol=0.25)


def test_shapes_and_companion():
    y, *_ = simulate_var1(T=200)
    model = BVAR(y, lags=4)
    draws = model.sample_posterior(3, seed=1)
    assert len(draws) == 3
    k, p = 2, 4
    m = k * p + 1
    for d in draws:
        assert isinstance(d, PosteriorDraw)
        assert d.Pi.shape == (k, m)
        assert d.Sigma.shape == (k, k)
        # Sigma symmetric PD
        assert np.allclose(d.Sigma, d.Sigma.T)
        assert np.all(np.linalg.eigvalsh(d.Sigma) > 0)
        F = d.companion()
        assert F.shape == (k * p, k * p)
        assert np.allclose(F[:k, :], d.Pi[:, : k * p])
        assert np.allclose(F[k:, : k * (p - 1)], np.eye(k * (p - 1)))
    u = model.residuals(draws[0])
    assert u.shape == (200 - p, k)
    # residuals consistent with layout: Y - X Pi'
    assert np.allclose(u, model.Y - model.X @ draws[0].Pi.T)


def test_dummies_handled():
    T = 500
    dummies = np.zeros((T, 2))
    dummies[100:110, 0] = 1.0
    dummies[300:305, 1] = 1.0
    dummy_coefs = np.array([[5.0, 0.0], [0.0, -4.0]])
    y, *_ = simulate_var1(T=T, seed=3, dummies=dummies, dummy_coefs=dummy_coefs)
    model = BVAR(y, lags=1, dummies=dummies)
    assert model.m == 2 * 1 + 1 + 2
    draws = model.sample_posterior(200, seed=7)
    Pi_mean = np.mean([d.Pi for d in draws], axis=0)
    assert Pi_mean.shape == (2, 5)
    D_hat = Pi_mean[:, 3:]  # dummy block after lags + constant
    # signs of the additive shifts should be recovered
    assert D_hat[0, 0] > 1.0
    assert D_hat[1, 1] < -1.0


def test_reproducibility_with_seed():
    y, *_ = simulate_var1(T=300, seed=5)
    model = BVAR(y, lags=2)
    d1 = model.sample_posterior(5, seed=123)
    d2 = model.sample_posterior(5, seed=123)
    d3 = model.sample_posterior(5, seed=124)
    for a, b in zip(d1, d2):
        assert np.array_equal(a.Pi, b.Pi)
        assert np.array_equal(a.Sigma, b.Sigma)
    assert not np.allclose(d1[0].Pi, d3[0].Pi)


def test_input_validation():
    y, *_ = simulate_var1(T=100)
    with pytest.raises(ValueError):
        BVAR(y[:, 0])  # 1-D
    with pytest.raises(ValueError):
        BVAR(y, lags=1, dummies=np.zeros((50, 1)))  # wrong length


# ---------------------------------------------------------------------------
# Dummy-initial-observation (single-unit-root) prior
# ---------------------------------------------------------------------------

def test_theta_default_off_matches_previous_behaviour():
    y, *_ = simulate_var1(T=300, seed=2)
    m_off = BVAR(y, lags=2)
    m_none = BVAR(y, lags=2, theta=None)
    np.testing.assert_allclose(m_off.B_post, m_none.B_post)
    np.testing.assert_allclose(m_off.S_post, m_none.S_post)
    assert m_off.df_post == m_none.df_post


def test_theta_adds_single_dummy_row_with_correct_layout():
    T = 300
    dummies = np.zeros((T, 2))
    dummies[50:55, 0] = 1.0
    y, *_ = simulate_var1(T=T, seed=4)
    theta = 2.5
    p = 3
    model = BVAR(y, lags=p, dummies=dummies, theta=theta)
    base = BVAR(y, lags=p, dummies=dummies)
    Yd, Xd = model._prior_dummies(model.scales)
    Yd0, Xd0 = base._prior_dummies(base.scales)
    assert Yd.shape[0] == Yd0.shape[0] + 1
    ybar = y[:p].mean(axis=0)
    np.testing.assert_allclose(Yd[-1], ybar / theta)
    k = y.shape[1]
    for l in range(p):
        np.testing.assert_allclose(Xd[-1, l * k:(l + 1) * k], ybar / theta)
    np.testing.assert_allclose(Xd[-1, k * p], 1.0 / theta)
    # zeros on the covid-dummy columns
    np.testing.assert_allclose(Xd[-1, k * p + 1:], 0.0)
    # degrees of freedom go up by one and the posterior changes
    assert model.df_post == base.df_post + 1
    assert not np.allclose(model.B_post, base.B_post)


def test_tight_theta_enforces_dummy_initial_observation_restriction():
    # As theta -> 0 the single dummy row is enforced exactly:
    # ybar = (sum_l A_l) ybar + c, i.e. the model admits ybar as a no-shock
    # steady state (the single-unit-root restriction).
    y, *_ = simulate_var1(T=200, seed=8)
    p = 1
    ybar = y[:p].mean(axis=0)

    def restriction_gap(model):
        A = model.B_post[:2, :].T  # k x k lag block (p = 1)
        c = model.B_post[2, :]
        return np.linalg.norm(ybar - (A @ ybar + c))

    loose = BVAR(y, lags=p, theta=100.0)
    tight = BVAR(y, lags=p, theta=1e-4)
    assert restriction_gap(tight) < restriction_gap(loose)
    assert restriction_gap(tight) < 1e-4 * max(np.linalg.norm(ybar), 1.0)


# ---------------------------------------------------------------------------
# Marginal likelihood and hyperparameter optimization
# ---------------------------------------------------------------------------

def test_log_marginal_likelihood_finite_and_sensitive_to_lam():
    y, *_ = simulate_var1(T=250, seed=9)
    vals = []
    for lam in (0.05, 0.2, 1.0):
        model = BVAR(y, lags=2, lam=lam, theta=1.0)
        v = model.log_marginal_likelihood()
        assert np.isfinite(v)
        vals.append(v)
    assert len(set(np.round(vals, 6))) == 3  # lam actually matters


def simulate_near_rw(T=250, seed=0):
    """Random-walk-like VAR(1): coefficients match the Minnesota prior mean."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.98, 0.0], [0.0, 0.97]])
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + 0.1 * rng.standard_normal(2)
    return y


def simulate_far_from_prior(T=250, seed=0):
    """Stationary VAR(1) with large cross coefficients (far from RW prior)."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.2, 0.65], [-0.55, 0.2]])
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = A @ y[t - 1] + 0.1 * rng.standard_normal(2)
    return y


def test_ml_prefers_looser_lam_for_large_coefficients():
    from boe_var.bvar import optimize_hyperparameters
    lam_grid = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    one = np.array([1.0])
    kw = dict(lags=1, mu_grid=one, theta_grid=one, refine=False,
              lam_grid=lam_grid)
    lam_rw = optimize_hyperparameters(simulate_near_rw(seed=13), **kw)["lam"]
    lam_far = optimize_hyperparameters(
        simulate_far_from_prior(seed=13), **kw)["lam"]
    # comparative ordering only: looser (larger) lam when the truth is far
    # from the random-walk prior, tighter when it is near the prior
    assert lam_far > lam_rw


def test_optimize_hyperparameters_returns_grid_argmax():
    from boe_var.bvar import optimize_hyperparameters
    y = simulate_near_rw(T=150, seed=21)
    lam_grid = np.array([0.1, 0.5])
    mu_grid = np.array([0.5, 2.0])
    theta_grid = np.array([0.5, 2.0])
    out = optimize_hyperparameters(y, lags=1, lam_grid=lam_grid,
                                   mu_grid=mu_grid, theta_grid=theta_grid,
                                   refine=False)
    best = -np.inf
    for lam in lam_grid:
        for mu in mu_grid:
            for th in theta_grid:
                v = BVAR(y, lags=1, lam=lam, mu=mu,
                         theta=th).log_marginal_likelihood()
                if v > best:
                    best, args = v, (lam, mu, th)
    assert (out["lam"], out["mu"], out["theta"]) == args
    np.testing.assert_allclose(out["log_ml"], best)


def test_optimize_hyperparameters_refine_does_not_worsen():
    from boe_var.bvar import optimize_hyperparameters
    y = simulate_near_rw(T=150, seed=22)
    grid = np.array([0.2, 1.0])
    coarse = optimize_hyperparameters(y, lags=1, lam_grid=grid,
                                      mu_grid=grid, theta_grid=grid,
                                      refine=False)
    refined = optimize_hyperparameters(y, lags=1, lam_grid=grid,
                                       mu_grid=grid, theta_grid=grid,
                                       refine=True)
    assert refined["log_ml"] >= coarse["log_ml"] - 1e-9


def test_scalar_marginal_likelihood_sequential_predictive_identity():
    """log p(Y) must equal the sum of one-step Student-t predictive log
    densities under a FIXED dummy-observation prior (k=1 conjugate NIG).

    This is an exact identity of the closed-form marginal likelihood: with
    prior dummies (Yd, Xd) held fixed, p(Y_1..T | prior) factorizes into
    sequential predictives y_t | y_<t ~ t_nu(x_t' bhat,
    s2 (1 + x_t' (Xa'Xa)^{-1} x_t)) with nu = n_a - m, s2 = SSR_a / nu,
    computed from the prior dummies plus previous observations. Verifies the
    normalizing constants (multigammaln, determinant, pi powers) exactly.
    Runs in milliseconds.
    """
    from scipy.stats import t as student_t

    rng = np.random.default_rng(5)
    T = 40
    y = np.zeros((T, 1))
    for i in range(1, T):
        y[i] = 0.7 * y[i - 1] + 0.3 + 0.5 * rng.standard_normal()

    model = BVAR(y, lags=2, lam=0.4, mu=1.5, theta=2.0)
    lml = model.log_marginal_likelihood()

    Yd, Xd = model._prior_dummies(model.scales)
    Ya, Xa = Yd.copy(), Xd.copy()
    total = 0.0
    for i in range(model.Y.shape[0]):
        x_t = model.X[i]
        XtX = Xa.T @ Xa
        bhat = np.linalg.solve(XtX, Xa.T @ Ya)
        resid = Ya - Xa @ bhat
        ssr = (resid.T @ resid).item()
        nu = Xa.shape[0] - model.m
        assert nu > 0
        scale2 = (ssr / nu) * (1.0 + (x_t @ np.linalg.solve(XtX, x_t)).item())
        total += student_t.logpdf(
            model.Y[i].item(), df=nu, loc=(x_t @ bhat).item(),
            scale=np.sqrt(scale2))
        Ya = np.vstack([Ya, model.Y[i][None, :]])
        Xa = np.vstack([Xa, x_t[None, :]])

    assert lml == pytest.approx(total, abs=1e-8)

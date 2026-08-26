"""Does the SVAR get the DIRECTION right?

The other suites check the estimator, the identification algebra and the
paper's published statistics. This one asks the economic question: when a
structural shock hits, does the economy move the way theory says it should?

For a sign-identified SVAR that question needs care, because **most of the
headline signs are imposed, not estimated**. `world_demand` raises UK GDP on
impact because SIGN_RESTRICTIONS says a draw is rejected unless it does.
Testing those tells you the sampler works -- worth doing, and done below --
but it is not evidence about the economy.

The evidence lives in what the restrictions leave FREE:

  * unrestricted variables. `eri` carries no restriction under any shock, and
    `oil_price` and `bank_rate` are free under several.
  * unrestricted horizons. Every restriction here is an IMPACT restriction, so
    h >= 1 is free for every variable including the restricted ones.

Those are the responses this file is really about.

The statistic is the posterior probability of a sign, not the median. A
Bayesian SVAR's answer to "does the Bank tighten?" is a distribution; taking
the median and calling it the direction throws away exactly the information
that says whether the direction is pinned down at all. Several responses here
turn out NOT to be pinned down, and those are pinned as such so nobody quotes
them as findings.

Measured at 4000 draws (291 accepted); thresholds sit well inside the
measured values so ordinary sampling noise does not flip a test.
"""

import numpy as np
import pandas as pd
import pytest

from boe_var.analysis import (
    _companion,
    estimated_shocks,
    fevd,
    historical_decomposition,
    irf,
)
from boe_var.bvar import BVAR
from boe_var.data import load_data
from boe_var.identification import (
    SHOCKS,
    SIGN_RESTRICTIONS,
    VARIABLES,
    ZERO_RESTRICTIONS,
    identify,
)

N_DRAWS = 4000
SAMPLE_SEED = 1
IDENT_SEED = 2
LAGS = 4
HORIZONS = 21

V = {v: i for i, v in enumerate(VARIABLES)}
S = {s: j for j, s in enumerate(SHOCKS)}


def _covid_dummies(index):
    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    D = np.zeros((len(index), len(quarters)))
    for j, q in enumerate(quarters):
        D[:, j] = (index == q).astype(float)
    return D


class _Fit:
    def __init__(self, model, pairs, dummies, index):
        self.model = model
        self.pairs = pairs
        self.dummies = dummies
        self.index = index
        self.irfs = np.stack([irf(d, B, HORIZONS) for d, B in pairs])  # (n,var,shock,H)

    def p_positive(self, shock, var, h=0):
        """Posterior probability that the response is positive."""
        return float((self.irfs[:, V[var], S[shock], h] > 0).mean())

    def p_negative(self, shock, var, h=0):
        return float((self.irfs[:, V[var], S[shock], h] < 0).mean())


@pytest.fixture(scope="module")
def fit():
    df = load_data()
    df = df.loc[
        (df.index >= pd.Period("1992Q1", "Q")) & (df.index <= pd.Period("2023Q2", "Q"))
    ]
    dummies = _covid_dummies(df.index)
    model = BVAR(
        df.to_numpy(dtype=float), lags=LAGS, dummies=dummies, lam=0.2, mu=1.0, theta=1.0
    )
    draws = model.sample_posterior(N_DRAWS, seed=SAMPLE_SEED)
    accepted = identify(
        draws, rng=np.random.default_rng(IDENT_SEED), compute_weights=False
    )
    pairs = [(d, B) for d, B, _ in accepted]
    assert len(pairs) >= 150, (
        f"only {len(pairs)} accepted draws; sign probabilities unstable"
    )
    return _Fit(model, pairs, dummies, df.index)


# ---------------------------------------------------------------------------
# A. The identification binds. Necessary before any sign below means anything.
# ---------------------------------------------------------------------------


def test_every_sign_restriction_holds_in_every_accepted_draw(fit):
    """A draw that violates its own restriction should never have been accepted."""
    bad = []
    for n, (_, B) in enumerate(fit.pairs):
        for shock, restrictions in SIGN_RESTRICTIONS.items():
            for var_i, expected in restrictions:
                got = B[var_i, S[shock]]
                if np.sign(got) != expected and abs(got) > 1e-12:
                    bad.append(f"draw {n}: {shock}->{VARIABLES[var_i]} = {got:.4g}")
    assert not bad, f"{len(bad)} sign-restriction violations, e.g. {bad[:3]}"


def test_every_zero_restriction_holds_in_every_accepted_draw(fit):
    """The zero block is exact algebra, not an approximation."""
    worst = 0.0
    for _, B in fit.pairs:
        for shock, rows in ZERO_RESTRICTIONS.items():
            for var_i in rows:
                worst = max(worst, abs(B[var_i, S[shock]]))
    assert worst < 1e-8, f"largest supposedly-zero impact response is {worst:.3g}"


def test_the_two_unidentified_shocks_carry_no_restrictions(fit):
    """They are residual directions, so their sign carries no meaning.

    Pinned so nobody reads an economic story into `unident_global` or
    `unident_uk`: nothing orients them, and the sign of their responses is an
    artefact of the rotation the sampler happened to draw.
    """
    for shock in ("unident_global", "unident_uk"):
        assert SIGN_RESTRICTIONS[shock] == []
    assert ZERO_RESTRICTIONS["unident_global"] == []


def test_the_exchange_rate_is_unrestricted_under_every_shock(fit):
    """`eri` is the one variable no restriction ever touches.

    That makes it the cleanest read on what the DATA say, and it is why the
    eri tests below are about evidence rather than about the sampler.
    """
    for shock in SHOCKS:
        restricted = {v for v, _ in SIGN_RESTRICTIONS[shock]} | set(
            ZERO_RESTRICTIONS[shock]
        )
        assert V["eri"] not in restricted, f"eri became restricted under {shock}"


# ---------------------------------------------------------------------------
# B. Free impact responses that match theory. Nothing here is imposed.
# ---------------------------------------------------------------------------


def test_world_demand_boom_raises_oil_prices(fit):
    """Unrestricted, and the textbook sign: more world activity bids up oil."""
    assert fit.p_positive("world_demand", "oil_price") >= 0.75


def test_world_demand_boom_raises_bank_rate(fit):
    """The strongest free result in the model: an endogenous policy response.

    Nothing ties Bank Rate to a world demand shock -- the restriction set
    leaves index 3 free under every world shock -- yet the posterior puts
    ~0.97 on a rise. World demand raises UK inflation and the Bank leans
    against it. That is the model earning a sign rather than being handed one.
    """
    assert fit.p_positive("world_demand", "bank_rate") >= 0.85


def test_world_demand_boom_appreciates_sterling(fit):
    """Free, and consistent with the rate response above."""
    assert fit.p_positive("world_demand", "eri") >= 0.70


def test_favourable_uk_supply_lowers_energy_prices(fit):
    """`cpi_energy` is unrestricted under UK shocks; it moves with headline CPI."""
    assert fit.p_negative("uk_supply", "cpi_energy") >= 0.65


# ---------------------------------------------------------------------------
# C. Free DYNAMICS. Every restriction is an impact restriction, so h >= 1 is
#    the model talking, not the identification.
# ---------------------------------------------------------------------------


def test_monetary_tightening_depresses_gdp_beyond_the_impact_quarter(fit):
    """h=0 is imposed; h=1..8 are not, and the sign holds anyway.

    This is the single most important free-dynamics result here: the
    contraction persists for two years rather than unwinding the quarter after
    the restriction stops binding, which is what a mis-identified monetary
    shock usually looks like.
    """
    for h in range(1, 9):
        p = fit.p_negative("uk_monpol", "uk_gdp", h)
        assert p >= 0.80, f"P(uk_gdp < 0 | uk_monpol) = {p:.2f} at h={h}"


def test_monetary_tightening_keeps_prices_down_beyond_impact(fit):
    """No price puzzle: CPI stays below baseline across the free horizons."""
    for h in range(1, 9):
        p = fit.p_negative("uk_monpol", "cpisa", h)
        assert p >= 0.80, f"P(cpisa < 0 | uk_monpol) = {p:.2f} at h={h}"


def test_uk_demand_boost_fades_rather_than_compounding(fit):
    """Positive throughout, but decaying -- a demand shock should not build."""
    ps = [fit.p_positive("uk_demand", "uk_gdp", h) for h in range(9)]
    assert all(p >= 0.70 for p in ps), ps
    assert ps[8] < ps[1], f"the demand effect is not fading: {ps[1]:.2f} -> {ps[8]:.2f}"


# ---------------------------------------------------------------------------
# D. Limits, pinned. Each of these is a real property of the fitted model that
#    a reader could easily mistake for a finding.
# ---------------------------------------------------------------------------


def test_the_exchange_rate_response_to_uk_shocks_is_undetermined(fit):
    """LIMIT (pinned). The model cannot tell you what sterling does.

    `eri` is unrestricted, so this is the data speaking -- and the data barely
    speak: P(appreciation) is 0.57-0.66 across all three UK shocks, which is
    close enough to a coin flip that no direction should be quoted, including
    the textbook one for a monetary tightening.

    Pinned rather than skipped: if a future data vintage or prior sharpens
    this, the test fails and the claim can be upgraded deliberately.
    """
    for shock in ("uk_demand", "uk_supply", "uk_monpol"):
        p = fit.p_positive(shock, "eri")
        assert 0.35 <= p <= 0.75, (
            f"eri under {shock} is now pinned at P(+)={p:.2f} -- if the "
            "posterior has sharpened, upgrade this to a directional test"
        )


def test_favourable_world_supply_and_energy_shocks_still_raise_bank_rate(fit):
    """LIMIT (pinned). Against the usual prior for a favourable supply shock.

    Both shocks lower world CPI and UK CPI on impact by construction, so a
    central bank reacting to inflation would ease. The posterior says tighten:
    P(+) ~0.81 for world_supply and ~0.75 for world_energy.

    This is not necessarily wrong -- both shocks also raise UK output, and a
    rule that weighs activity can tighten into a favourable supply shock -- but
    it is the opposite of what most readers will assume, so it is pinned here
    rather than left to be discovered in a chart.
    """
    assert fit.p_positive("world_supply", "bank_rate") >= 0.65
    assert fit.p_positive("world_energy", "bank_rate") >= 0.60


def test_the_posterior_is_not_stationary(fit):
    """LIMIT (pinned). Almost every draw has a companion root at or above one.

    Measured: max |eigenvalue| has median 1.006 and runs to 1.034, and 289 of
    291 accepted draws sit at or above 1. That is expected for a levels VAR
    under a Minnesota prior centred on random walks, and it is why the IRFs do
    not die out.

    The consequence is what matters: long-horizon responses and the FEVD at
    long horizons are not convergent objects here. Read them at the short
    horizons the paper reports, not at h=20.
    """
    eig = np.array(
        [np.abs(np.linalg.eigvals(_companion(d))).max() for d, _ in fit.pairs]
    )
    share = float((eig >= 1.0).mean())
    assert share >= 0.80, (
        f"only {share:.0%} of draws are non-stationary (was ~99%) -- if the "
        "prior or data changed, the long-horizon caveat may no longer apply"
    )
    assert eig.max() < 1.10, f"a draw is materially explosive: max root {eig.max():.3f}"


# ---------------------------------------------------------------------------
# E. Internal consistency. Cheap, and they catch the failures that make every
#    sign above meaningless.
# ---------------------------------------------------------------------------


def test_variance_shares_sum_to_one_at_every_horizon(fit):
    shares = np.stack([fevd(d, B, HORIZONS) for d, B in fit.pairs])
    assert np.allclose(shares.sum(axis=2), 1.0, atol=1e-8)


def test_variance_shares_are_probabilities(fit):
    shares = np.stack([fevd(d, B, HORIZONS) for d, B in fit.pairs])
    assert shares.min() >= -1e-12 and shares.max() <= 1.0 + 1e-12


def test_no_impulse_response_is_nan_or_infinite(fit):
    """A NaN here silently voids the posterior rather than failing loudly.

    That is not hypothetical: commit 44cb504 fixed exactly such a NaN. This
    guards the whole array rather than the statistic that happened to expose
    it last time.
    """
    assert np.isfinite(fit.irfs).all()


def test_structural_shocks_are_standardised(fit):
    """Unit-variance shocks are what makes an IRF a response to "one s.d."."""
    draw, B = fit.pairs[0]
    shocks = estimated_shocks(draw, B, fit.model.residuals(draw))
    variances = shocks.var(axis=0)
    assert np.all(variances > 0.5) and np.all(variances < 2.0), variances


def test_impulse_responses_are_linear_in_the_shock(fit):
    """A VAR is linear: flipping a shock's sign must flip its whole path.

    If this fails, "the direction of the effect" is not a well-defined
    question and nothing else in this file can be trusted.
    """
    draw, B = fit.pairs[0]
    assert np.allclose(irf(draw, -B, 6), -irf(draw, B, 6))


def test_the_historical_decomposition_reconciles_to_the_data(fit):
    """Shock contributions plus the deterministic parts must rebuild the data.

    The docstring states the identity -- shocks + covid + deterministic =
    data -- and it is the check that the attribution actually accounts for
    the series rather than merely resembling it. If this drifts, every
    "what drove GDP" statement built on the decomposition is wrong by the
    size of the drift.
    """
    draw, B = fit.pairs[0]
    residuals = fit.model.residuals(draw)
    hd = historical_decomposition(draw, B, residuals, dummies=fit.dummies[LAGS:])
    y = fit.model.y[LAGS:]
    deterministic = y - hd["stochastic"] - hd["covid"]
    rebuilt = hd["shocks"].sum(axis=2) + hd["covid"] + deterministic
    assert np.allclose(rebuilt, y, atol=1e-8), (
        f"decomposition misses the data by up to {np.abs(rebuilt - y).max():.3g}"
    )

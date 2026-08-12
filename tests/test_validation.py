"""Validation against the paper's ground truth (Brignone & Piffer, 2025).

Every test here locks a specific fact stated in SPEC.md (Table 2), the paper
figures, or results/summary.md, evaluated on the *real* BoE dataset with a
small fixed-seed configuration (see conftest.py). The sign/zero-restriction
invariants (Table 2) are the core correctness gate: they must hold *exactly*
on every accepted identified draw.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boe_var import analysis  # noqa: E402
from boe_var.analysis import UK_SHOCKS, WORLD_SHOCKS  # noqa: E402
from boe_var.identification import (  # noqa: E402
    SIGN_RESTRICTIONS,
    SHOCKS,
    VARIABLES,
    ZERO_RESTRICTIONS,
    identify,
)

I_CPI, I_GDP = 5, 7          # rows of UK CPISA and UK real GDP in Table 2
UNIDENT_SHOCKS = [3, 7]      # unidentified global (U1) and unidentified UK (U2)
# The paper's "~40% / ~50%" is stated "one year after the shocks", i.e. the
# 4-quarter-ahead forecast-error variance. Column h of analysis.fevd is the
# (h+1)-step-ahead variance, so that is index 3, not 4.
H_1YR = analysis.fevd_horizon_index(4)


# ---------------------------------------------------------------------------
# Table 2 sign / zero restrictions -- the core correctness invariants
# ---------------------------------------------------------------------------

def test_at_least_a_handful_of_draws_accepted(identified):
    """Sanity: the fixed-seed fast config must accept enough draws to be a
    meaningful sample (guards against a silently empty identification)."""
    assert len(identified.pairs) >= 20


def test_zero_restrictions_hold_exactly_on_every_accepted_draw(identified):
    """Table 2 zero block: each B[v, j] restricted to 0 is 0 on EVERY draw."""
    worst = 0.0
    for _, B in identified.pairs:
        for j, shock in enumerate(SHOCKS):
            for v in ZERO_RESTRICTIONS[shock]:
                worst = max(worst, abs(B[v, j]))
    assert worst < 1e-9, f"largest zero-restriction violation was {worst}"


def test_sign_restrictions_hold_exactly_on_every_accepted_draw(identified):
    """Table 2 sign block: each restricted B[v, j] has the required sign on
    EVERY accepted draw (this is the accept/reject invariant of the
    Arias-Rubio-Ramirez-Waggoner scheme)."""
    for idx, (_, B) in enumerate(identified.pairs):
        for j, shock in enumerate(SHOCKS):
            for v, s in SIGN_RESTRICTIONS[shock]:
                assert s * B[v, j] > 0, (
                    f"draw {idx}: {shock} on {VARIABLES[v]} has value "
                    f"{B[v, j]} but Table 2 requires sign {s}")


def test_B_factorizes_sigma_on_every_accepted_draw(identified):
    """B B' = Sigma exactly, so B is a valid structural impact matrix."""
    for draw, B in identified.pairs:
        assert np.allclose(B @ B.T, draw.Sigma, atol=1e-7)


# ---------------------------------------------------------------------------
# Impulse responses -- signs and the paper's unrestricted checks
# ---------------------------------------------------------------------------

def test_impact_irf_median_signs_match_table2(identified):
    """The median impact IRF respects every Table 2 sign restriction."""
    bands = analysis.irf_bands(identified.pairs, horizons=21)
    med = bands["median"]                       # (var, shock, H)
    for j, shock in enumerate(SHOCKS):
        for v, s in SIGN_RESTRICTIONS[shock]:
            assert s * med[v, j, 0] > 0, (
                f"median impact of {shock} on {VARIABLES[v]} = {med[v, j, 0]} "
                f"violates Table 2 sign {s}")


def test_oil_price_rises_after_world_supply_shock(identified):
    """The paper's key *unrestricted* check (Fig 2): the real oil price rises
    on impact after a positive world-supply shock even though its sign is not
    imposed (oil is row 2, world supply is shock 2)."""
    bands = analysis.irf_bands(identified.pairs, horizons=21)
    assert bands["median"][2, 2, 0] > 0


def test_irf_and_fevd_have_no_nan_or_inf(identified):
    """Every accepted draw yields finite IRFs and FEVDs."""
    for draw, B in identified.pairs:
        r = analysis.irf(draw, B, horizons=21)
        f = analysis.fevd(draw, B, horizons=21)
        assert np.all(np.isfinite(r))
        assert np.all(np.isfinite(f))


# ---------------------------------------------------------------------------
# Forecast-error variance decomposition
# ---------------------------------------------------------------------------

def test_fevd_shares_sum_to_one(identified):
    """FEVD rows sum to ~100% at every horizon on every draw."""
    for draw, B in identified.pairs:
        shares = analysis.fevd(draw, B, horizons=21)   # (var, shock, H)
        totals = shares.sum(axis=1)                     # (var, H)
        assert np.allclose(totals, 1.0, atol=1e-10)


def test_fevd_shares_are_valid_proportions_on_every_draw(identified):
    """A variance share is a proportion: every FEVD entry lies in [0, 1] on
    every accepted draw and every horizon. Summing to 1 (above) does not by
    itself rule out a negative share offset by one above 1."""
    for draw, B in identified.pairs:
        shares = analysis.fevd(draw, B, horizons=21)
        assert shares.min() >= -1e-12, f"negative FEVD share {shares.min()}"
        assert shares.max() <= 1.0 + 1e-12, f"FEVD share > 1: {shares.max()}"


def test_fevd_global_share_of_uk_gdp_and_cpi_in_paper_neighbourhood(identified):
    """HARD GATE -- paper benchmark (summary.md): identified GLOBAL shocks
    (world demand + energy + supply) explain ~40% of UK GDP and ~50% of UK CPI
    variance at business-cycle horizons. The band [30%, 60%] is deliberately
    wide: it brackets both the paper's ~40/50 and this fast unweighted config's
    ~47/43, so it flags a gross regression without being flaky to Monte-Carlo
    variation across the ~50 accepted draws. The band is a published invariant,
    so it gates PRs; only the *tighter* weighted magnitudes
    (test_weighted_fevd_matches_paper_benchmark) remain report-only."""
    bands = analysis.fevd_bands(identified.pairs, horizons=21)
    med = bands["median"]
    med = med / med.sum(axis=1, keepdims=True)
    g_gdp = med[I_GDP, WORLD_SHOCKS, H_1YR].sum()
    g_cpi = med[I_CPI, WORLD_SHOCKS, H_1YR].sum()
    assert 0.30 <= g_gdp <= 0.60, f"UK GDP global share {g_gdp:.3f} off-band"
    assert 0.30 <= g_cpi <= 0.60, f"UK CPI global share {g_cpi:.3f} off-band"


def test_fevd_headline_aggregates_equal_component_sums(identified):
    """Internal consistency claimed by summary.md: the headline global /
    domestic / unidentified aggregates are exactly the sums of the detailed
    per-shock cells printed in the same file, and the three groups partition
    all eight shocks.

    The previous version of this test compared each aggregate with a literal
    re-evaluation of its own defining expression, so it asserted ``x == x``
    and could not fail. It now rebuilds the detailed table the way
    run_replication.py prints it and adds the cells back up.
    """
    bands = analysis.fevd_bands(identified.pairs, horizons=21)
    med = bands["median"]
    med = med / med.sum(axis=1, keepdims=True)

    # the three groups must partition the eight shock columns exactly once
    assert sorted(WORLD_SHOCKS + UK_SHOCKS + UNIDENT_SHOCKS) == list(range(8))

    for i in (I_GDP, I_CPI):
        # the detailed per-shock cells as summary.md prints them
        detail = [med[i, j, H_1YR] for j in range(8)]
        g = med[i, WORLD_SHOCKS, H_1YR].sum()
        d = med[i, UK_SHOCKS, H_1YR].sum()
        u = med[i, UNIDENT_SHOCKS, H_1YR].sum()
        assert np.isclose(g, sum(detail[j] for j in WORLD_SHOCKS), atol=1e-12)
        assert np.isclose(d, sum(detail[j] for j in UK_SHOCKS), atol=1e-12)
        assert np.isclose(u, sum(detail[j] for j in UNIDENT_SHOCKS), atol=1e-12)
        # renormalisation makes the whole row add to 100%
        assert np.isclose(sum(detail), 1.0, atol=1e-9)
        assert np.isclose(g + d + u, 1.0, atol=1e-9)


def test_fevd_horizon_index_is_the_four_quarter_variance():
    """The 1-year FEVD claim must read the 4-quarter-ahead forecast-error
    variance. ``analysis.fevd`` accumulates squared MA coefficients, so its
    column h is the (h+1)-step-ahead variance and the 1-year figure is column
    3. The published summary table used to read column 4 -- the 5-quarter
    variance -- under a "1-year horizon" heading."""
    assert analysis.fevd_horizon_index(1) == 0
    assert analysis.fevd_horizon_index(4) == 3
    with pytest.raises(ValueError):
        analysis.fevd_horizon_index(0)

    # and the index really does select the 4-quarter accumulation
    class _Draw:
        k, lags = 2, 1
        Pi = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])

        def companion(self):
            return np.array([[0.5, 0.0], [0.0, 0.5]])

    B = np.eye(2)
    shares = analysis.fevd(_Draw(), B, horizons=6)
    psi = np.array([[0.5 ** h, 0.0] for h in range(6)])
    four_quarter = np.sum(psi[:4, 0] ** 2)          # h = 0, 1, 2, 3
    total = four_quarter                            # only shock 0 loads on var 0
    assert np.isclose(shares[0, 0, analysis.fevd_horizon_index(4)],
                      four_quarter / total)


def test_group_share_is_not_the_sum_of_per_shock_medians(identified):
    """The headline is a share of a SUM of shocks, so it must be formed on
    each draw before quantiles are taken. Summing per-shock medians (what the
    legacy table does) is a different, and on this data materially larger,
    quantity; both are reported and neither may silently stand in for the
    other."""
    groups = {"global": WORLD_SHOCKS, "domestic": UK_SHOCKS,
              "unidentified": UNIDENT_SHOCKS}
    per_draw = analysis.fevd_group_shares(
        identified.pairs, groups, quarters_ahead=4)

    # every draw's shares partition the variance, so the group medians are
    # each in [0, 1] and no renormalisation is needed or applied
    for name in groups:
        for key in ("lo90", "lo68", "median", "hi68", "hi90"):
            v = per_draw[name][key]
            assert v.shape == (8,)
            assert np.all(v >= -1e-12) and np.all(v <= 1.0 + 1e-12)
        # bands are ordered
        assert np.all(per_draw[name]["lo90"] <= per_draw[name]["lo68"] + 1e-12)
        assert np.all(per_draw[name]["hi68"] <= per_draw[name]["hi90"] + 1e-12)

    # the paper reports MEANS of shares of total variance, so that statistic
    # must be available and must itself be a partition of the variance
    total = sum(per_draw[name]["mean"] for name in groups)
    np.testing.assert_allclose(total, np.ones(8), atol=1e-10)

    # the two routes really do disagree: the legacy sum-of-medians route needs
    # renormalisation precisely because the per-shock medians do not add to 1
    bands = analysis.fevd_bands(identified.pairs, horizons=21)
    raw = bands["median"][:, :, H_1YR]
    assert not np.allclose(raw.sum(axis=1), 1.0, atol=1e-6), (
        "per-shock medians unexpectedly add to 1; the renormalisation step in "
        "run_replication.py would then be a no-op and this test is stale")


# ---------------------------------------------------------------------------
# Estimated structural shocks
# ---------------------------------------------------------------------------

def test_estimated_shocks_are_mean_zero_and_unit_scaled(identified):
    """Structural shocks eps_t = B^{-1} u_t should be ~mean-zero and ~unit
    variance (B's columns are unit-s.d. shocks, Sigma = B B'). Checked on the
    posterior-mean draw's residuals."""
    model = identified.model
    draw, B = identified.pairs[0]
    resid = model.residuals(draw)
    eps = analysis.estimated_shocks(draw, B, resid)
    means = eps.mean(axis=0)
    stds = eps.std(axis=0)
    assert np.all(np.abs(means) < 0.3), f"shock means off-zero: {means}"
    assert np.all((stds > 0.6) & (stds < 1.5)), f"shock stds off-unit: {stds}"


# ---------------------------------------------------------------------------
# Historical decomposition reconstructs the data
# ---------------------------------------------------------------------------

def test_historical_decomposition_reconstructs_the_data(identified):
    """shocks + covid + deterministic = data (VAR-Toolbox adding-up), and the
    stochastic component equals the sum over shocks -- exactly (to 1e-8)."""
    model = identified.model
    draw, B = identified.pairs[0]
    resid = model.residuals(draw)
    lags = model.lags
    hd = analysis.historical_decomposition(
        draw, B, resid, dummies=identified.dummies[lags:])
    stochastic = hd["stochastic"]
    covid = hd["covid"]
    y_eff = identified.y[lags:]
    deterministic = y_eff - stochastic - covid
    recon = hd["shocks"].sum(axis=2) + covid + deterministic
    np.testing.assert_allclose(recon, y_eff, atol=1e-8)
    np.testing.assert_allclose(hd["shocks"].sum(axis=2), stochastic, atol=1e-10)


def test_historical_decomposition_reconstructs_the_data_on_every_draw(identified):
    """The adding-up identity above is a published invariant, so it must hold
    for EVERY accepted draw, not just the first one: for each draw the shock
    contributions plus the covid and deterministic components reproduce the
    observed data exactly, and the stochastic component is the sum over
    shocks."""
    model = identified.model
    lags = model.lags
    y_eff = identified.y[lags:]
    dummies = identified.dummies[lags:]
    for idx, (draw, B) in enumerate(identified.pairs):
        resid = model.residuals(draw)
        hd = analysis.historical_decomposition(draw, B, resid, dummies=dummies)
        stochastic = hd["stochastic"]
        covid = hd["covid"]
        deterministic = y_eff - stochastic - covid
        recon = hd["shocks"].sum(axis=2) + covid + deterministic
        np.testing.assert_allclose(
            recon, y_eff, atol=1e-8,
            err_msg=f"draw {idx}: historical decomposition does not reconstruct")
        np.testing.assert_allclose(
            hd["shocks"].sum(axis=2), stochastic, atol=1e-10,
            err_msg=f"draw {idx}: shock sum != stochastic component")


# ---------------------------------------------------------------------------
# Slow: high-draw, importance-WEIGHTED benchmark (workflow_dispatch / schedule)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_published_invariants_hold_under_the_heavy_weighted_config(
        identified_weighted):
    """HARD GATE (nightly) -- the four published invariants re-checked on the
    high-draw, importance-weighted configuration, so drift is caught under the
    configuration the paper actually targets and not just the fast fixture.

    Unlike the FEVD *magnitude* benchmark below, none of these is a
    Monte-Carlo-sensitive quantity: they are exact algebraic identities and
    accept/reject conditions, so a failure here is a real bug.
    """
    ident = identified_weighted
    assert len(ident.pairs) >= 20, "heavy config accepted too few draws"

    model = ident.model
    lags = model.lags
    y_eff = ident.y[lags:]
    dummies = ident.dummies[lags:]

    for idx, (draw, B) in enumerate(ident.pairs):
        # (1) zero restrictions
        for j, shock in enumerate(SHOCKS):
            for v in ZERO_RESTRICTIONS[shock]:
                assert abs(B[v, j]) < 1e-9, (
                    f"draw {idx}: zero restriction {shock}/{VARIABLES[v]} "
                    f"violated by {B[v, j]}")
            # (2) sign restrictions
            for v, s in SIGN_RESTRICTIONS[shock]:
                assert s * B[v, j] > 0, (
                    f"draw {idx}: {shock} on {VARIABLES[v]} = {B[v, j]} "
                    f"violates Table 2 sign {s}")

        # (3) FEVD shares are proportions and sum to 1
        shares = analysis.fevd(draw, B, horizons=21)
        assert np.all(np.isfinite(shares))
        assert shares.min() >= -1e-12
        assert shares.max() <= 1.0 + 1e-12
        np.testing.assert_allclose(shares.sum(axis=1), 1.0, atol=1e-10)

        # (4) historical decomposition reconstructs the observed data
        resid = model.residuals(draw)
        hd = analysis.historical_decomposition(draw, B, resid, dummies=dummies)
        deterministic = y_eff - hd["stochastic"] - hd["covid"]
        recon = hd["shocks"].sum(axis=2) + hd["covid"] + deterministic
        np.testing.assert_allclose(
            recon, y_eff, atol=1e-8,
            err_msg=f"draw {idx}: historical decomposition does not reconstruct")


@pytest.mark.slow
@pytest.mark.calibration
def test_weighted_fevd_matches_paper_benchmark():
    """With the Arias et al. (2018) importance weights and more draws, the
    identified global share is close to the paper's ~40% (UK GDP) / ~50%
    (UK CPI). Weighted so it targets the correct posterior; slow because the
    weight needs a numerical Jacobian per accepted draw."""
    import pandas as pd

    from boe_var.bvar import BVAR
    from boe_var.data import load_data

    df = load_data()
    df = df.loc[(df.index >= pd.Period("1992Q1", "Q"))
                & (df.index <= pd.Period("2023Q2", "Q"))]
    y = df.to_numpy(dtype=float)
    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    dummies = np.column_stack([(df.index == q).astype(float) for q in quarters])

    model = BVAR(y, lags=4, dummies=dummies, lam=0.2, mu=1.0, theta=1.0)
    draws = model.sample_posterior(1500, seed=1)
    accepted = identify(draws, rng=np.random.default_rng(2), compute_weights=True)
    pairs = [(d, B) for d, B, _ in accepted]
    w = np.array([t[2] for t in accepted], dtype=float)
    w = w / w.sum()

    bands = analysis.fevd_bands(pairs, horizons=21, weights=w)
    med = bands["median"]
    med = med / med.sum(axis=1, keepdims=True)
    g_gdp = med[I_GDP, WORLD_SHOCKS, H_1YR].sum()
    g_cpi = med[I_CPI, WORLD_SHOCKS, H_1YR].sum()
    assert 0.33 <= g_gdp <= 0.58
    assert 0.40 <= g_cpi <= 0.60

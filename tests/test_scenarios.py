"""Monetary-policy shock scenarios: size, sign and shape.

A scenario here is the forecast from the current data plus one structural
shock at the first forecast quarter, minus the unconditional forecast. The
SVAR is linear, so three properties must hold exactly on every accepted
draw, for any size and sign:

  * zero shock = baseline;
  * the deviation equals size x the impulse response from ``analysis.irf``
    (the forecast recursion and the IRF code agree);
  * symmetry and proportionality: +s and -s give mirror-image paths, 2s
    gives twice the path.

The shape checks after that are properties of the fitted posterior, stated
as posterior probabilities with thresholds well inside the measured values
(measured on the fast ``identified`` fixture, 43 accepted draws: 0.88, 0.84,
0.95, 0.79). They pin what the model says. They are NOT a numerical
validation against Brignone & Piffer (2025) Figures 2-3, which needs the
authors' numbers (issue #11).
"""

import numpy as np
import pytest

from boe_var.analysis import SHOCK_NAMES, irf
from boe_var.forecast import _iterate, unconditional_forecast

MP = SHOCK_NAMES.index("UK mon. pol.")
BANK_RATE, CPI, WORLD_GDP, UK_GDP = 3, 5, 0, 7
H = 21
SIZES = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def scenario(draw, B, y, size, shock=MP, horizons=H):
    """Forecast path with ``size`` s.d. of structural ``shock`` at h=0."""
    shocks = np.zeros((horizons, draw.k))
    shocks[0] = size * np.asarray(B)[:, shock]
    return _iterate(draw, y, horizons, shocks=shocks)


@pytest.fixture(scope="module")
def setup(identified):
    y = identified.y
    pairs = identified.pairs
    base = [unconditional_forecast(d, y, horizons=H) for d, _ in pairs]
    return y, pairs, base


def test_zero_shock_is_the_baseline(setup):
    y, pairs, base = setup
    for (d, B), b in zip(pairs, base):
        np.testing.assert_array_equal(scenario(d, B, y, 0.0), b)


def test_scenario_deviation_is_size_times_the_impulse_response(setup):
    y, pairs, base = setup
    for (d, B), b in zip(pairs, base):
        resp = irf(d, B, H)[:, MP, :].T          # (H, k)
        for s in SIZES:
            np.testing.assert_allclose(scenario(d, B, y, s) - b, s * resp,
                                       atol=1e-9, rtol=1e-9)


def test_tightening_and_easing_are_mirror_images(setup):
    y, pairs, base = setup
    for (d, B), b in zip(pairs, base):
        for s in (0.5, 1.0, 2.0):
            up = scenario(d, B, y, s) - b
            down = scenario(d, B, y, -s) - b
            np.testing.assert_allclose(up, -down, atol=1e-9)
            np.testing.assert_allclose(scenario(d, B, y, 2 * s) - b, 2 * up,
                                       atol=1e-9)


def test_a_rate_rise_of_given_size_scales_exactly(setup):
    """Normalise to the Bank Rate impact: a 100bp shock is four 25bp shocks,
    and the impact on Bank Rate is the size asked for."""
    y, pairs, base = setup
    for (d, B), b in zip(pairs, base):
        impact = np.asarray(B)[BANK_RATE, MP]
        assert impact > 0, "the MP shock raises Bank Rate on impact (imposed)"
        dev25 = scenario(d, B, y, 0.25 / impact) - b
        dev100 = scenario(d, B, y, 1.00 / impact) - b
        assert dev25[0, BANK_RATE] == pytest.approx(0.25)
        np.testing.assert_allclose(dev100, 4 * dev25, atol=1e-9)


@pytest.fixture(scope="module")
def mp_irfs(identified):
    return np.stack([irf(d, B, H)[:, MP, :] for d, B in identified.pairs])


def test_bank_rate_response_is_persistent_but_decays(mp_irfs):
    """Bank Rate stays above baseline after five years (unit-root posterior)
    but below its impact rise: the model does not revert within the window."""
    br = mp_irfs[:, BANK_RATE, :]
    assert (br[:, 20] < br[:, 0]).mean() >= 0.70
    assert (br[:, 20] > 0).mean() >= 0.65


def test_the_price_level_effect_builds_rather_than_fading(mp_irfs):
    """No price puzzle and no quick reversal: CPI at h=12 is further below
    baseline than on impact."""
    cpi = mp_irfs[:, CPI, :]
    assert (cpi[:, 12] < cpi[:, 0]).mean() >= 0.65


def test_a_uk_rate_shock_barely_moves_world_gdp(mp_irfs):
    """Small open economy: at two years the world GDP response is smaller
    than the UK GDP response. Only the impact response is restricted to zero."""
    w, uk = np.abs(mp_irfs[:, WORLD_GDP, 8]), np.abs(mp_irfs[:, UK_GDP, 8])
    assert (w < uk).mean() >= 0.80


def test_gdp_trough_comes_after_the_impact_quarter(mp_irfs):
    gdp = mp_irfs[:, UK_GDP, :13]
    assert (np.argmin(gdp, axis=1) > 0).mean() >= 0.60

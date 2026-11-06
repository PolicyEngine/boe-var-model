"""Held-out check of the band scale factors (scripts/run_coverage_evaluation.py).

The committed factors k = z(nominal)/z(empirical) are fitted on the same
origins their coverage is judged on. These tests pin the machinery that
scores them on origins they were not fitted on.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    from run_coverage_evaluation import (  # noqa: E402
        band_stretch,
        holdout_calibration,
        run_coverage,
    )
finally:
    sys.path.pop(0)

Z68, Z90 = 0.994458, 1.644854


def test_band_stretch_is_the_edge_multiple_on_the_outturn_side():
    med = np.array([0.0, 0.0, 0.0, 1.0])
    lo = np.array([-1.0, -2.0, -1.0, 1.0])
    hi = np.array([2.0, 1.0, 1.0, 1.0])
    actual = np.array([3.0, -1.0, 0.0, 2.0])
    np.testing.assert_allclose(band_stretch(med, lo, hi, actual),
                               [1.5, 0.5, 0.0, np.inf])


def _synthetic(n_o=400, H=2, k=2, true_sd=2.0, seed=0):
    """Bands built for sd 1 around a correct median, outturns with sd 2."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_o, H, k)) * true_sd
    return {68: np.abs(e) / Z68, 90: np.abs(e) / Z90}


def test_under_covering_bands_are_repaired_out_of_sample():
    out = holdout_calibration(_synthetic())
    for lv in ("68", "90"):
        nominal = out[lv]["nominal"]
        raw = out[lv]["raw"]["mean"]
        assert raw < nominal - 0.15, "bands built for sd 1 must under-cover"
        for scheme in ("in_sample_calibrated", "leave_one_out_calibrated",
                       "rolling_origin_calibrated"):
            assert out[lv][scheme]["mean"] == pytest.approx(nominal, abs=0.04)


def test_in_sample_is_optimistic_when_the_error_scale_shifts():
    """A calibration fitted on calm origins and applied to a volatile spell is
    exactly what an in-sample check cannot see: in-sample says nominal,
    rolling-origin shows the shortfall."""
    S = _synthetic(n_o=200, true_sd=1.0, seed=1)
    later = _synthetic(n_o=100, true_sd=3.0, seed=2)
    S = {lv: np.concatenate([S[lv], later[lv]]) for lv in S}
    out = holdout_calibration(S)
    r = out["68"]
    assert r["in_sample_calibrated"]["mean"] == pytest.approx(0.68, abs=0.04)
    assert r["rolling_origin_calibrated"]["mean"] < 0.6


def test_rolling_origin_uses_only_observed_outturns():
    out = holdout_calibration(_synthetic(n_o=30, H=3), min_train=12)
    scored = out["68"]["rolling_origin_scored_origins_by_horizon"]
    # origin i can train on origins 0..i-h-1, so the first scored origin
    # at horizon h (1-based) is i = 12 + h - 1
    assert scored == {"h1": 30 - 12, "h2": 30 - 13, "h3": 30 - 14}


def test_recorded_stretch_reproduces_hits_on_real_data():
    from boe_var.data import load_data

    y = load_data().to_numpy(float)
    T, k = y.shape
    rec = {}
    hits, n = run_coverage(y, lags=4, horizons=3, first_origin=T - 3 - 6,
                           n_draws=20, n_paths=2, seed=0, record=rec)
    for lv in (68, 90):
        S = rec["stretch"][lv]
        assert S.shape == (6, 3, k)
        np.testing.assert_array_equal((S <= 1.0).sum(axis=0), hits[lv])


def _artifact():
    import json

    path = Path(__file__).resolve().parents[1] / "results" / "coverage_evaluation.json"
    return json.loads(path.read_text())


def test_artifact_records_the_evaluated_window():
    d = _artifact()
    assert d["evaluated_origins"] == {
        "first": "2012Q1", "last": "2024Q1", "last_outturn": "2026Q1"}


@pytest.mark.parametrize("spec", ["headline", "production_spec"])
def test_68_calibration_survives_out_of_sample(spec):
    """Recorded 6 Nov 2026: rolling-origin 0.696 / 0.686, LOO 0.681 / 0.684."""
    b = _artifact()["calibration_holdout"][spec]["68"]
    assert b["rolling_origin_calibrated"]["mean"] == pytest.approx(0.68, abs=0.03)
    assert b["leave_one_out_calibrated"]["mean"] == pytest.approx(0.68, abs=0.03)


@pytest.mark.parametrize("spec", ["headline", "production_spec"])
def test_90_bands_still_under_cover_after_calibration(spec):
    """LIMIT (pinned). A Gaussian half-width factor cannot fix the tails:
    rolling-origin calibrated 90% coverage is 0.840 / 0.822, short of 0.90.
    This is evidence for issue #12 (fat tails / volatility), not a bug."""
    b = _artifact()["calibration_holdout"][spec]["90"]
    roll = b["rolling_origin_calibrated"]["mean"]
    assert roll < 0.87
    assert roll > b["rolling_origin_raw_same_origins"]["mean"]
    assert b["in_sample_calibrated"]["mean"] < 0.88, (
        "even in sample the Gaussian k does not reach 90%")

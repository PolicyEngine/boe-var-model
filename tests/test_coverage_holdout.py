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
        "first": "2012Q1", "last": "2024Q2", "last_outturn": "2026Q2"}


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


# --- Conformal recalibration (issue #12) ----------------------------------


def _conf():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import run_coverage_evaluation as m  # noqa: E402
    finally:
        sys.path.pop(0)
    return m


def test_conformal_factor_is_the_finite_sample_rank():
    m = _conf()
    scores = np.arange(1, 13, dtype=float)          # n = 12
    # ceil(13 * 0.9) = 12 -> the largest; ceil(13 * 0.68) = 9 -> the 9th
    assert m.conformal_factor(scores, 0.9) == (12.0, False)
    assert m.conformal_factor(scores, 0.68) == (9.0, False)
    # ceil(13 * 0.95) = 13 > n: capped at the max, and flagged
    assert m.conformal_factor(scores, 0.95) == (12.0, True)


def _fat_tailed(n_o=600, H=2, k=2, seed=0):
    """Bands built for a unit Gaussian, outturns 1.5 x Student-t(3): too
    narrow AND fat-tailed. The Gaussian map z(nominal)/z(empirical) assumes
    the shape is right and only the scale is wrong, so it under-shoots."""
    rng = np.random.default_rng(seed)
    e = rng.standard_t(3, (n_o, H, k)) * 1.5
    return {68: np.abs(e) / Z68, 90: np.abs(e) / Z90}


def test_conformal_fixes_fat_tails_where_the_gaussian_factor_cannot():
    m = _conf()
    S = _fat_tailed()
    K90, _ = m.conformal_rolling_factors(S[90], 0.9)
    G90 = m.gaussian_rolling_factors(S[90], 90)
    ok = ~np.isnan(K90)
    conf = (S[90] <= K90)[ok].mean()
    gaus = (S[90] <= G90)[ok].mean()
    assert conf == pytest.approx(0.90, abs=0.025)
    assert gaus < 0.87, "the Gaussian map under-shoots a fat tail"


def test_conformal_repairs_gaussian_under_coverage_too():
    m = _conf()
    S = _synthetic()
    for lv in (68, 90):
        K, _ = m.conformal_rolling_factors(S[lv], lv / 100.0)
        ok = ~np.isnan(K)
        assert (S[lv] <= K)[ok].mean() == pytest.approx(lv / 100.0, abs=0.04)


def test_no_leakage_perturbing_later_origins_leaves_earlier_bands_unchanged():
    """The factor for origin i at horizon h may use only origins j <= i-h-1.
    Blowing up every origin from j0 onward -- including origin i0 - h itself,
    the first one whose outturn is NOT yet observed at i0 -- must leave every
    factor at origins <= i0 identical."""
    m = _conf()
    base = _synthetic(n_o=60, H=4, k=3, seed=5)[90]
    K0, _ = m.conformal_rolling_factors(base, 0.9)
    i0 = 40
    for h in range(4):
        j0 = i0 - h                     # first origin unseen at i0, horizon h
        pert = base.copy()
        pert[j0:, h, :] *= 50.0
        K1, _ = m.conformal_rolling_factors(pert, 0.9)
        np.testing.assert_array_equal(K1[: i0 + 1, h], K0[: i0 + 1, h])
        # and the perturbation is visible one origin later (the test bites)
        assert not np.array_equal(K1[i0 + 1:, h], K0[i0 + 1:, h])


def test_no_leakage_end_to_end_on_real_data():
    """Perturb the LAST outturn quarter of the data and rerun the whole
    pipeline: every conformal band at an origin that could not have seen it
    must be unchanged."""
    from boe_var.data import load_data

    m = _conf()
    y = load_data().to_numpy(float)
    T, k = y.shape
    H, n_o = 2, 8
    first = T - H - n_o
    recs = []
    for yy in (y, np.vstack([y[:-1], y[-1:] * 3.0 + 10.0])):
        rec = {}
        m.run_coverage(yy, lags=4, horizons=H, first_origin=first,
                       n_draws=10, n_paths=2, seed=0, record=rec)
        recs.append(rec)
    for lv in (68, 90):
        K0, _ = m.conformal_rolling_factors(recs[0]["stretch"][lv], lv / 100,
                                            min_train=2)
        K1, _ = m.conformal_rolling_factors(recs[1]["stretch"][lv], lv / 100,
                                            min_train=2)
        # data row T-1 is the h-step outturn of origin index n_o-1-h; it is
        # first observable at origin index n_o (beyond the window), so every
        # scored factor must be identical
        np.testing.assert_array_equal(K0, K1)
        assert not np.array_equal(recs[0]["stretch"][lv][-1],
                                  recs[1]["stretch"][lv][-1])


def test_recalibrated_68_90_edges_match_the_band_factors():
    """The full-distribution recalibration uses the same factors at the
    published band edges, so its 68%/90% coverage is the conformal one."""
    m = _conf()
    rng = np.random.default_rng(7)
    n_o, H, k = 80, 2, 2
    taus = np.asarray(m.TAUS)
    from statistics import NormalDist
    zq = np.array([NormalDist().inv_cdf(t) for t in taus])
    Q = np.broadcast_to(zq, (n_o, H, k, taus.size)).copy()
    y = rng.standard_t(3, (n_o, H, k))
    Qr, fac = m.recalibrated_quantiles(Q, y)
    S90 = m.band_stretch(Q[..., 10], Q[..., 1], Q[..., 19], y)
    K90, _ = m.conformal_rolling_factors(S90, 0.9)
    ok = ~np.isnan(K90)
    np.testing.assert_allclose(fac[0.9][ok], K90[ok])
    inside = (Qr[..., 1] <= y) & (y <= Qr[..., 19])
    np.testing.assert_array_equal(inside[ok], (S90 <= K90)[ok])
    assert np.all(np.diff(Qr[ok], axis=-1) >= 0)


def test_quantile_crps_matches_the_gaussian_closed_form():
    m = _conf()
    from statistics import NormalDist
    taus = np.linspace(0.0005, 0.9995, 2001)
    Q = np.array([NormalDist().inv_cdf(t) for t in taus])
    # CRPS(N(0,1), y=0) = 2 phi(0) - 1/sqrt(pi) = 0.23369
    assert float(m.quantile_crps(Q, 0.0, taus)) == pytest.approx(0.23369, abs=2e-3)


@pytest.mark.parametrize("spec", ["headline", "production_spec",
                                  "production_spec_covid_outturns_excluded"])
def test_conformal_does_not_fix_the_90_band_out_of_sample(spec):
    """LIMIT (pinned, issue #12). Recorded 6 Nov 2026, production spec:
    rolling-origin 90% coverage conformal 0.824 vs Gaussian k 0.822 (raw
    0.768), mean half-width factor 1.77 vs 1.44. The misses cluster in time,
    so calibrating on past errors -- parametrically or not -- lags them.
    If this starts failing because coverage reached ~0.9, revisit adopting
    the conformal factors."""
    b = _artifact()["conformal_calibration"][spec]["90"]
    assert b["conformal"]["mean"] < 0.86
    assert b["conformal"]["mean"] > b["raw"]["mean"]
    assert b["capped_share"] == 0.0


def test_conformal_68_band_is_no_better_than_the_gaussian_factor():
    b = _artifact()["conformal_calibration"]["production_spec"]["68"]
    assert b["conformal"]["mean"] == pytest.approx(0.68, abs=0.03)
    assert b["conformal"]["mean"] <= b["gaussian_k"]["mean"] + 0.005


def test_conformal_block_scores_the_same_origins_as_the_gaussian_holdout():
    d = _artifact()
    c = d["conformal_calibration"]["production_spec"]["90"]
    g = d["calibration_holdout"]["production_spec"]["90"]
    assert c["scored_origins_by_horizon"] == \
        g["rolling_origin_scored_origins_by_horizon"]
    assert c["gaussian_k"]["mean"] == pytest.approx(
        g["rolling_origin_calibrated"]["mean"], abs=1e-4)

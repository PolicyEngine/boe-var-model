"""Year-on-year coverage scoring beyond four quarters (issue #27).

The committed-artefact tests pin that year-on-year coverage equals level
coverage at h1-h4, where the year-ago level is observed. Beyond h4 the
year-ago level is itself a forecast, and nothing checked that arithmetic.
Here the simulated paths are captured and the year-on-year hits are
recomputed a second way, with ``forecast.yoy`` on each full path.
"""

import sys
from pathlib import Path

import numpy as np

from boe_var import forecast as fc
from boe_var.data import load_data

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _run_coverage():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import run_coverage_evaluation as mod
    finally:
        sys.path.pop(0)
    return mod


def test_yoy_hits_match_an_independent_computation_at_h5_to_h8(monkeypatch):
    mod = _run_coverage()
    y = load_data().to_numpy(float)
    T, k = y.shape
    horizons, n_origins, n_draws, n_paths = 8, 6, 20, 2
    first_origin = T - horizons - n_origins

    captured = []
    real = fc.sample_forecast

    def recording(draw, y_hist, *args, **kwargs):
        path = real(draw, y_hist, *args, **kwargs)
        captured.append((len(y_hist) - 1, path))
        return path

    monkeypatch.setattr(mod.fc, "sample_forecast", recording)
    yoy = {}
    hits, n = mod.run_coverage(
        y, lags=4, horizons=horizons, first_origin=first_origin,
        n_draws=n_draws, n_paths=n_paths, seed=3, yoy=yoy)
    assert (n == n_origins).all()
    assert len(captured) == n_origins * n_draws * n_paths

    expected = {68: np.zeros((horizons, k)), 90: np.zeros((horizons, k))}
    for origin in range(first_origin, first_origin + n_origins):
        # full history + path, then the four-quarter difference of the
        # whole thing; keep the forecast quarters
        paths = np.stack([
            fc.yoy(np.vstack([y[: origin + 1], p]))[-horizons:]
            for o, p in captured if o == origin])
        actual = fc.yoy(y[: origin + horizons + 1])[-horizons:]
        for lv, (a, b) in {68: (16.0, 84.0), 90: (5.0, 95.0)}.items():
            lo, hi = np.percentile(paths, [a, b], axis=0)
            expected[lv] += (lo <= actual) & (actual <= hi)

    for lv in (68, 90):
        np.testing.assert_array_equal(yoy["hits"][lv], expected[lv])
        # h1-h4: identical to the level hits; h5-h8: a different quantity
        np.testing.assert_array_equal(yoy["hits"][lv][:4], hits[lv][:4])
    assert any((yoy["hits"][lv][4:] != hits[lv][4:]).any() for lv in (68, 90))


def test_yoy_coverage_is_the_same_on_the_simple_rate():
    """The published fans are simple year-on-year rates; coverage is scored
    on log differences. The conversion is increasing, so an outturn is
    inside a converted band exactly when it is inside the log band. The
    band of converted paths equals the converted band up to the linear
    interpolation between two neighbouring draws (1e-5 here), not exactly."""
    rng = np.random.default_rng(0)
    paths = rng.normal(2.0, 3.0, size=(500, 8))
    actual = rng.normal(2.0, 3.0, size=8)
    s = fc.log_to_simple_pct
    for a, b in ((16.0, 84.0), (5.0, 95.0)):
        lo, hi = np.percentile(paths, [a, b], axis=0)
        np.testing.assert_array_equal(
            (lo <= actual) & (actual <= hi),
            (s(lo) <= s(actual)) & (s(actual) <= s(hi)))
        lo_s, hi_s = np.percentile(s(paths), [a, b], axis=0)
        np.testing.assert_allclose(lo_s, s(lo), atol=1e-5)
        np.testing.assert_allclose(hi_s, s(hi), atol=1e-5)

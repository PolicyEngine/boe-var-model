import numpy as np
import pytest

from boe_var.evaluation import benjamini_hochberg, rolling_origin_evaluation


def test_benjamini_hochberg_matches_the_textbook_step_up():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    q = benjamini_hochberg(p)
    m = p.size
    expected = np.minimum.accumulate(
        (p * m / np.arange(1, m + 1))[::-1])[::-1]
    np.testing.assert_allclose(q, np.minimum(expected, 1.0))
    assert np.all(np.diff(q) >= -1e-12), "q-values must be monotone in p"
    assert np.all(q >= p - 1e-12), "BH can only make a p-value less impressive"


def test_benjamini_hochberg_handles_missing_and_unsorted_input():
    q = benjamini_hochberg([0.5, None, 0.01, float("nan"), 0.2])
    assert np.isnan(q[1]) and np.isnan(q[3])
    finite = q[[0, 2, 4]]
    assert np.all(finite <= 1.0)
    # smallest p keeps the smallest q regardless of input order
    assert q[2] == finite.min()
    assert np.isnan(benjamini_hochberg([None, None])).all()


def test_benjamini_hochberg_is_not_fooled_by_the_smallest_of_many_nulls():
    """The reason this exists: with 64 independent null tests the smallest
    p-value is routinely near 0.01, and reporting it unadjusted reads as a
    discovery. BH must return a q-value that does not."""
    rng = np.random.default_rng(0)
    q = benjamini_hochberg(rng.uniform(size=64))
    assert q.min() > 0.10


def _stationary_ar(T=100, phi=0.35, seed=4):
    rng = np.random.default_rng(seed)
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = phi * y[t - 1] + rng.normal(scale=0.2, size=2)
    return y


def test_rolling_evaluation_is_json_safe_and_uses_only_valid_origins():
    report = rolling_origin_evaluation(
        _stationary_ar(), lags=1, horizons=3, first_origin=60,
        bvar_kwargs={"lam": 0.5, "mu": 1.0},
    )
    assert report["origins"] == 37
    assert [row["horizon"] for row in report["horizons"]] == [1, 2, 3]
    assert all(len(row["relative_rmse"]) == 2 for row in report["horizons"])
    assert report["horizons"][0]["variables_beating_random_walk"] >= 1


def test_rolling_evaluation_rejects_empty_windows():
    with pytest.raises(ValueError, match="no valid rolling origins"):
        rolling_origin_evaluation(
            _stationary_ar(T=30), lags=4, horizons=8, first_origin=25
        )


def test_rolling_evaluation_validates_dummy_alignment():
    y = _stationary_ar(T=60)
    with pytest.raises(ValueError, match="same rows"):
        rolling_origin_evaluation(y, lags=1, dummies=np.zeros((59, 2)))


def test_rolling_evaluation_runs_with_dummies_and_stays_finite():
    """The dummy path of the rolling evaluation is what the PUBLISHED model
    uses (six Covid quarters as exogenous regressors), yet until now no test
    exercised it and no committed artifact used it. Pin that it runs, keeps
    the same origin count, and produces finite ratios."""
    y = _stationary_ar(T=100)
    D = np.zeros((100, 2))
    D[70, 0] = 1.0
    D[71, 1] = 1.0
    plain = rolling_origin_evaluation(y, lags=1, horizons=3, first_origin=60)
    with_d = rolling_origin_evaluation(
        y, lags=1, horizons=3, first_origin=60, dummies=D
    )
    assert with_d["origins"] == plain["origins"]
    for row in with_d["horizons"]:
        assert np.isfinite(row["bvar_rmse"]).all()
        assert np.isfinite(row["relative_rmse_vs_drift"]).all()
    # The benchmarks ignore the dummies, so they must be untouched.
    for a, b in zip(plain["horizons"], with_d["horizons"]):
        np.testing.assert_allclose(a["drift_rmse"], b["drift_rmse"])
        np.testing.assert_allclose(a["random_walk_rmse"], b["random_walk_rmse"])


def test_dummy_columns_after_the_origin_do_not_change_the_forecast():
    """Real-time safety of the Covid-dummy specification.

    A dummy for a quarter LATER than the forecast origin is identically zero
    over the training sample. It must therefore leave the point forecast
    numerically unchanged -- otherwise adding the pandemic dummies to a
    rolling evaluation would leak information about quarters the forecaster
    has not seen. Compared here against a dummy matrix truncated to zero
    after each origin, which is unambiguously real-time.
    """
    y = _stationary_ar(T=100)
    late = np.zeros((100, 2))
    # last origin is T - horizons - 1 = 96, so these two quarters are never in
    # any training sample
    late[97, 0] = 1.0
    late[98, 1] = 1.0
    none = rolling_origin_evaluation(y, lags=1, horizons=3, first_origin=60)
    with_late = rolling_origin_evaluation(
        y, lags=1, horizons=3, first_origin=60, dummies=late
    )
    for a, b in zip(none["horizons"], with_late["horizons"]):
        np.testing.assert_allclose(
            a["bvar_errors_by_origin"], b["bvar_errors_by_origin"], atol=1e-10,
            err_msg="a not-yet-observed dummy column changed the forecast")


def test_diebold_mariano_signs_and_symmetry():
    """Negative statistic must favour the model, and swapping the two error
    series must flip the sign while leaving the p-value unchanged."""
    import numpy as np
    from boe_var.evaluation import _diebold_mariano

    rng = np.random.default_rng(0)
    good = rng.normal(scale=0.2, size=(49, 3))
    bad = rng.normal(scale=1.0, size=(49, 3))

    stat_a, p_a = _diebold_mariano(good, bad, 1)
    stat_b, p_b = _diebold_mariano(bad, good, 1)

    assert (stat_a < 0).all(), "model with smaller errors must give a negative stat"
    assert (stat_b > 0).all()
    np.testing.assert_allclose(stat_a, -stat_b, rtol=1e-12)
    np.testing.assert_allclose(p_a, p_b, rtol=1e-12)
    assert (p_a < 0.01).all(), "a 5x accuracy gap over 49 origins must register"


def test_diebold_mariano_does_not_reject_when_accuracy_is_equal():
    """Two independent draws from the same distribution must not look
    significantly different -- the guard against a test that always fires."""
    import numpy as np
    from boe_var.evaluation import _diebold_mariano

    rng = np.random.default_rng(7)
    rejects = 0
    for _ in range(40):
        a = rng.normal(size=(49, 1))
        b = rng.normal(size=(49, 1))
        _, p = _diebold_mariano(a, b, 1)
        if p[0] < 0.05:
            rejects += 1
    assert rejects <= 6, f"rejected {rejects}/40 under the null; test is miscalibrated"


def test_ar1_cannot_diverge_from_the_drift_path():
    """The AR(1) benchmark must nest drift, not explode.

    Fitted with an intercept, the long-run per-step increment is c/(1-phi),
    which detonates as phi approaches 1 -- on the real data that produced a
    single -203 log-point path and made the model look 4x better than the
    benchmark. Mean-deviation form pins the long-run increment to the sample
    mean change, so the two paths must stay close even after a large shock.
    """
    import numpy as np
    from boe_var.evaluation import _ar1_forecast, _drift_forecast

    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0.1, 1.0, 80))[:, None]
    y[60] -= 25.0  # a Covid-sized collapse in the training tail

    gap = np.abs(_ar1_forecast(y, 8) - _drift_forecast(y, 8)).max()
    assert gap < 5.0, f"AR(1) ran {gap:.1f} away from the drift path"


def test_benchmarks_reproduce_a_pure_linear_trend():
    """Both naive benchmarks must be exact on a deterministic trend; if they
    are not, any ratio computed against them is measuring the benchmark."""
    import numpy as np
    from boe_var.evaluation import _ar1_forecast, _drift_forecast

    train = np.arange(40, dtype=float)[:, None] * 0.5 + 3.0
    target = np.arange(40, 44, dtype=float)[:, None] * 0.5 + 3.0
    for fn in (_drift_forecast, _ar1_forecast):
        np.testing.assert_allclose(fn(train, 4), target, atol=1e-9)


def test_worst_origin_mse_share_detects_a_single_dominating_origin():
    """The guard against publishing a ratio that is one observation."""
    import numpy as np
    from boe_var.evaluation import rolling_origin_evaluation

    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0, 1, (90, 2)), axis=0)
    rep = rolling_origin_evaluation(y, lags=2, horizons=2, first_origin=60)
    shares = rep["horizons"][0]["worst_origin_mse_share"]
    assert set(shares) == {"bvar", "random_walk", "drift", "ar1"}
    for vals in shares.values():
        assert all(0.0 <= v <= 1.0 for v in vals)
    # On well-behaved noise no single origin should dominate.
    assert max(shares["random_walk"]) < 0.5

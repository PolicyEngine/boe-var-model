import numpy as np
import pytest

from boe_var.evaluation import rolling_origin_evaluation


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

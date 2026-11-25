"""results/lambda_sensitivity.json: held-out accuracy and coverage at
different Minnesota tightness (issues #11, #12).

The file is a report: the published model keeps lam = 0.2. The tests make
sure it measures what it says. At lam = 0.2 the wrapper must reproduce the
committed coverage and rolling evaluations cell for cell, so the other
columns differ from the published model in lam and nothing else.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SCRIPTS = REPO / "scripts"
VARIABLES = ["world_gdp", "world_cpi", "oil_price", "bank_rate",
             "eri", "cpisa", "cpi_energy", "uk_gdp"]


def _mod():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import run_lambda_sensitivity as mod
    finally:
        sys.path.pop(0)
    return mod


@pytest.fixture(scope="module")
def report():
    return json.loads((RESULTS / "lambda_sensitivity.json").read_text())


def test_is_not_stale(report):
    assert report["code_version"] == _mod().code_version(), (
        "evaluation code or data changed without regenerating "
        "results/lambda_sensitivity.json: dispatch the CI workflow (job "
        "lambda-sensitivity) or run scripts/run_lambda_sensitivity.py")


def test_configuration_is_the_committed_coverage_evaluation(report):
    cov = json.loads((RESULTS / "coverage_evaluation.json").read_text())
    for key in ("draws_per_origin", "paths_per_draw", "seed", "lags",
                "horizons", "first_origin", "origins", "sample"):
        assert report[key] == cov[key], key
    assert report["lambdas"] == [0.2, 0.5, 1.04]
    assert report["base"] == "0.2"


def test_published_lambda_reproduces_the_committed_coverage(report):
    """The lam = 0.2 column is the published specification, so it must be
    the committed production-spec coverage, on levels and year on year."""
    cov = json.loads((RESULTS / "coverage_evaluation.json").read_text())
    prod = cov["production_spec"]
    run = report["runs"]["0.2"]
    for block in ("coverage", "coverage_yoy"):
        for lv in ("68", "90"):
            for h in range(1, 9):
                for var in VARIABLES:
                    assert run[block][lv][f"h{h}"][var] == pytest.approx(
                        prod[block][lv][f"h{h}"][var], abs=1e-9), (
                            block, lv, h, var)


def test_published_lambda_reproduces_the_committed_point_accuracy(report):
    roll = json.loads((RESULTS / "rolling_evaluation.json").read_text())
    rows = {r["horizon"]: r for r in roll["production_spec"]["horizons"]}
    run = report["runs"]["0.2"]["relative_rmse_vs_random_walk"]
    for h in range(1, 9):
        for var in VARIABLES:
            assert run[f"h{h}"][var] == pytest.approx(
                rows[h]["relative_rmse_by_variable"][var], rel=1e-6), (h, var)


def test_ratios_are_to_the_published_lambda(report):
    base = report["runs"]["0.2"]
    for var in VARIABLES:
        assert base["width90_ratio_to_base"][var] == 1.0
        assert base["crps_ratio_to_base"][var] == 1.0
    for k, run in report["runs"].items():
        for h in range(1, 9):
            for var in VARIABLES:
                assert run["point_rmse_ratio_to_base"][f"h{h}"][var] == (
                    pytest.approx(run["point_rmse"][f"h{h}"][var]
                                  / base["point_rmse"][f"h{h}"][var],
                                  abs=1e-4))


def test_a_looser_prior_gives_wider_bands(report):
    """Less shrinkage means more parameter uncertainty: the 90% band is
    wider for every variable at 0.5 than at 0.2, and wider again at 1.04."""
    r = report["runs"]
    for var in VARIABLES:
        assert 1.0 < r["0.5"]["width90_ratio_to_base"][var] \
            < r["1.04"]["width90_ratio_to_base"][var], var


def test_reduced_scale_run_and_that_the_swap_is_undone():
    mod = _mod()
    original = mod.cov.BVAR
    from boe_var.data import load_data
    T = len(load_data())
    rep = mod.build_report(lambdas=(0.2, 1.04), n_draws=8,
                           first_origin=T - 4 - 5, horizons=4)
    assert mod.cov.BVAR is original
    assert rep["origins"] == 5
    assert set(rep["runs"]) == {"0.2", "1.04"}
    for run in rep["runs"].values():
        for lv in ("68", "90"):
            vals = [v for h in run["coverage_yoy"][lv].values()
                    for v in h.values()]
            assert all(0.0 <= v <= 1.0 for v in vals)
        assert np.isfinite(list(run["crps_ratio_to_base"].values())).all()
    assert rep["runs"]["0.2"] != rep["runs"]["1.04"]
    text = mod.markdown(rep)
    assert "| 1.04 |" in text and "## UK CPI" in text


def test_the_swap_changes_only_lambda(monkeypatch):
    """coverage_at(lam) must build exactly BVAR(..., lam=lam)."""
    mod = _mod()
    seen = []
    real = mod.BVAR

    class Spy(real):
        def __init__(self, *a, **kw):
            seen.append(kw.get("lam"))
            super().__init__(*a, **kw)

    monkeypatch.setattr(mod, "BVAR", Spy)
    from boe_var.data import load_data
    df = load_data()
    y = df.to_numpy(float)
    mod.coverage_at(0.7, y, mod.covid_dummies(df.index), n_draws=4,
                    first_origin=len(y) - 4 - 2, horizons=4)
    assert seen and set(seen) == {0.7}

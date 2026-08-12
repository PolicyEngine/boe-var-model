"""Gates on the committed real-data evaluation artifacts in ``results/``.

The site quotes headline numbers from ``rolling_evaluation.json``,
``coverage_evaluation.json`` and ``unemployment_satellite_validation.json``. Replication is
CI-gated elsewhere, but these artifacts are committed JSONs -- without the
tests below, a regression in evaluation code on real data (or a silent edit
to the artifacts) would ship without any red build.

Three layers:

1. **Headline assertions** (fast, hermetic): the recorded numbers the site
   quotes must still be in the files, within a small tolerance.
2. **Regeneration smoke** (~seconds, real packaged data): the evaluation
   machinery re-run at reduced scale must produce finite, well-formed,
   directionally sane output.
3. **Staleness guard**: each artifact records ``code_version`` -- a SHA-256
   over the evaluation-relevant sources written by its runner script. If
   evaluation code changes without regenerating the artifact, the recorded
   hash no longer matches and the test fails. Regeneration is cheap
   (each runner finishes in a few seconds): re-run the script and commit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boe_var.data import load_data
from boe_var.evaluation import evaluation_code_version, rolling_origin_evaluation

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
SCRIPTS = REPO / "scripts"

VARIABLES = ["world_gdp", "world_cpi", "oil_price", "bank_rate",
             "eri", "cpisa", "cpi_energy", "uk_gdp"]


def _load(name):
    return json.loads((RESULTS / name).read_text())


# --------------------------------------------------------------------------
# 1. Headline assertions on the committed artifacts
# --------------------------------------------------------------------------

class TestRollingEvaluationArtifact:
    """results/rolling_evaluation.json -- the point-forecast skill table."""

    # Snapshot of the published headline ratios (model RMSE / benchmark
    # RMSE). Regenerating on unchanged data reproduces these bit-for-bit;
    # the tolerance only absorbs BLAS-level float noise across machines.
    RTOL = 1e-3
    H1_VS_RW = {"world_gdp": 1.0450, "world_cpi": 0.7329, "oil_price": 1.0301,
                "bank_rate": 0.8809, "eri": 1.0386, "cpisa": 0.6293,
                "cpi_energy": 0.9762, "uk_gdp": 1.0637}
    H1_VS_DRIFT = {"world_gdp": 1.0710, "world_cpi": 0.9433,
                   "oil_price": 1.0135, "bank_rate": 0.7933, "eri": 1.0299,
                   "cpisa": 0.8306, "cpi_energy": 0.9771, "uk_gdp": 1.0635}
    H1_VS_AR1 = {"world_gdp": 0.7227, "world_cpi": 0.9713,
                 "oil_price": 1.0199, "bank_rate": 1.3822, "eri": 1.0680,
                 "cpisa": 1.0284, "cpi_energy": 0.9802, "uk_gdp": 0.6984}
    H8_VS_RW = {"world_gdp": 0.7108, "world_cpi": 0.6741, "oil_price": 1.0837,
                "bank_rate": 1.0280, "eri": 1.4104, "cpisa": 0.6731,
                "cpi_energy": 1.1343, "uk_gdp": 1.0581}

    def test_design_matches_the_published_evaluation(self):
        report = _load("rolling_evaluation.json")
        assert report["origins"] == 49
        assert report["first_origin"] == 80
        assert report["last_origin"] == 128
        assert report["lags"] == 4
        assert report["variables"] == VARIABLES
        assert [row["horizon"] for row in report["horizons"]] == list(range(1, 9))
        assert all(row["n_origins"] == 49 for row in report["horizons"])
        assert report["finite"] is True

    @pytest.mark.parametrize("key, expected", [
        ("relative_rmse_by_variable", H1_VS_RW),
        ("relative_rmse_vs_drift_by_variable", H1_VS_DRIFT),
        ("relative_rmse_vs_ar1_by_variable", H1_VS_AR1),
    ])
    def test_h1_relative_rmse_matches_recorded_values(self, key, expected):
        row = _load("rolling_evaluation.json")["horizons"][0]
        for var, value in expected.items():
            assert row[key][var] == pytest.approx(value, rel=self.RTOL), (
                f"{key}[{var}] drifted from the published value"
            )

    def test_h8_relative_rmse_vs_rw_matches_recorded_values(self):
        row = _load("rolling_evaluation.json")["horizons"][7]
        for var, value in self.H8_VS_RW.items():
            assert row["relative_rmse_by_variable"][var] == pytest.approx(
                value, rel=self.RTOL
            )

    def test_win_counts_vs_random_walk_match_recorded_values(self):
        report = _load("rolling_evaluation.json")
        counts = [row["variables_beating_random_walk"]
                  for row in report["horizons"]]
        assert counts == [4, 4, 4, 3, 3, 3, 3, 3]

    def test_all_ratios_are_finite_and_plausible(self):
        report = _load("rolling_evaluation.json")
        for row in report["horizons"]:
            for key in ("relative_rmse_by_variable",
                        "relative_rmse_vs_drift_by_variable",
                        "relative_rmse_vs_ar1_by_variable"):
                vals = np.array(list(row[key].values()))
                assert np.isfinite(vals).all()
                assert ((vals > 0.1) & (vals < 10)).all()


class TestProductionSpecEvaluationArtifact:
    """The like-for-like evaluation of the specification actually published.

    The headline block above estimates without the six Covid dummies; every
    published forecast estimates with them. These are the numbers a reader
    should quote when asking how the *published* forecaster performs, so they
    are pinned separately.
    """

    RTOL = 1e-3
    # UK GDP / CPI / Bank Rate against the random walk WITH DRIFT, the
    # benchmark the model has to beat to be worth running.
    H1_VS_DRIFT = {"uk_gdp": 0.9873, "cpisa": 0.8232, "bank_rate": 0.7809}
    H8_VS_DRIFT = {"uk_gdp": 0.9469, "cpisa": 1.0117, "bank_rate": 0.8298}

    def test_block_is_present_and_shares_the_headline_design(self):
        report = _load("rolling_evaluation.json")
        block = report["production_spec"]
        assert block["covid_dummies"] == "2020Q1-2021Q2"
        assert block["origins"] == report["origins"]
        assert [row["horizon"] for row in block["horizons"]] == list(range(1, 9))

    @pytest.mark.parametrize("h_index, expected", [(0, H1_VS_DRIFT),
                                                   (7, H8_VS_DRIFT)])
    def test_relative_rmse_vs_drift_matches_recorded_values(self, h_index,
                                                            expected):
        row = _load("rolling_evaluation.json")["production_spec"]["horizons"][h_index]
        for var, value in expected.items():
            assert row["relative_rmse_vs_drift_by_variable"][var] == pytest.approx(
                value, rel=self.RTOL)

    def test_covid_dummies_help_uk_gdp_and_barely_move_cpi(self):
        """The substantive finding this block exists to record: dummying out
        the pandemic quarters is what separates a UK GDP forecast that is
        worse than a drifting random walk from one that is not. CPI is
        essentially unaffected, so the CPI weakness is a property of the
        model, not of the Covid treatment."""
        report = _load("rolling_evaluation.json")
        for h in (0, 3, 7):
            base = report["horizons"][h]["relative_rmse_vs_drift_by_variable"]
            prod = (report["production_spec"]["horizons"][h]
                    ["relative_rmse_vs_drift_by_variable"])
            assert prod["uk_gdp"] < base["uk_gdp"] - 0.05, (
                "Covid dummies no longer materially improve the UK GDP ratio")
            assert abs(prod["cpisa"] - base["cpisa"]) < 0.05, (
                "Covid dummies now move CPI materially; the recorded reading "
                "that CPI weakness is Covid-independent is stale")

    def test_no_result_survives_multiplicity_control_against_drift(self):
        """The bluntest number in the artifact.

        The evaluation runs 64 Diebold-Mariano tests per benchmark. Against
        the random walk WITH DRIFT -- the only benchmark that is not trivially
        weak on trending log levels -- not one of them survives
        Benjamini-Hochberg control at any conventional level, in either the
        headline or the published specification. Individual unadjusted
        p-values around 0.02 therefore do not support a forecasting claim.
        """
        mult = _load("rolling_evaluation.json")["multiplicity"]
        for block in ("headline", "production_spec"):
            drift = mult[block]["drift"]
            assert drift["tests"] == 64
            assert drift["n_q_below_0.10"] == 0, (
                f"{block}: something now beats drift at FDR 10% (min q = "
                f"{drift['min_q']}); the 'no distinguishable skill' reading "
                "is stale and every consumer of it needs revisiting")

    def test_uk_gdp_is_still_not_significantly_better_than_drift(self):
        """Improved is not the same as skilful. Even under the published
        specification the UK GDP accuracy difference against a drifting random
        walk is not distinguishable from zero, and that must not be quietly
        lost when the ratios drop below 1."""
        block = _load("rolling_evaluation.json")["production_spec"]
        for h in (0, 3, 7):
            p = block["horizons"][h]["dm_pvalue_vs_drift_by_variable"]["uk_gdp"]
            assert p is None or p > 0.10, (
                f"UK GDP now beats drift at h={h + 1} with p={p}; the "
                "published 'no distinguishable GDP skill' reading is stale")


class TestCoverageEvaluationArtifact:
    """results/coverage_evaluation.json -- predictive-band calibration."""

    def test_design_matches_the_rolling_evaluation(self):
        report = _load("coverage_evaluation.json")
        assert report["origins"] == 49
        assert report["first_origin"] == 80
        assert report["lags"] == 4
        assert report["horizons"] == 8
        assert report["draws_per_origin"] == 100
        assert report["paths_per_draw"] == 5

    def test_68_band_mean_coverage_stays_in_the_recorded_band(self):
        """The site quotes ~57-71% empirical coverage for the 68% band
        (mean across the eight variables, per horizon)."""
        cov = _load("coverage_evaluation.json")["coverage"]["68"]
        means = [np.mean(list(cov[f"h{h}"].values())) for h in range(1, 9)]
        assert min(means) == pytest.approx(0.5663, abs=0.01)
        assert max(means) == pytest.approx(0.7117, abs=0.01)
        for h, m in enumerate(means, start=1):
            assert 0.55 <= m <= 0.72, f"68% band mean coverage at h{h}: {m}"

    def test_90_band_covers_more_than_the_68_band_everywhere(self):
        report = _load("coverage_evaluation.json")
        for h in range(1, 9):
            for var in VARIABLES:
                lo = report["coverage"]["68"][f"h{h}"][var]
                hi = report["coverage"]["90"][f"h{h}"][var]
                assert 0.0 <= lo <= hi <= 1.0

    def test_90_band_mean_coverage_stays_in_the_recorded_band(self):
        cov = _load("coverage_evaluation.json")["coverage"]["90"]
        means = [np.mean(list(cov[f"h{h}"].values())) for h in range(1, 9)]
        assert np.mean(means) == pytest.approx(0.772, abs=0.02)


class TestSatelliteValidationArtifact:
    """results/unemployment_satellite_validation.json -- the unemployment
    satellite's skill.

    The site's claim: fed by the VAR's own GDP forecasts (``svar``), the
    satellite regression (du on growth and lagged du, OLS 1992Q1-2025Q1,
    furlough quarters dummied out) beats no-change at horizons 1-4 (relative RMSE
    0.82-0.99, excluding furlough targets) and is clearly worse beyond,
    which is why the published path is capped at four quarters.
    """

    def test_design_matches_the_published_run(self):
        design = _load("unemployment_satellite_validation.json")["design"]
        assert design["origins_used"] == 73
        assert design["horizons"] == 8
        assert design["first_origin"] == "2005Q1"
        assert design["last_origin"] == "2023Q1"

    def test_svar_beats_no_change_at_h1_to_4_ex_covid(self):
        rows = _load("unemployment_satellite_validation.json")["ex_covid_targets"]
        ratios = {row["horizon"]: row["ratio_vs_rw"]["svar"] for row in rows}
        for h in (1, 2, 3, 4):
            assert 0.80 <= ratios[h] < 1.0, (
                f"site claims svar/rw in 0.82-0.99 at h{h}; got {ratios[h]}"
            )

    def test_svar_is_clearly_worse_beyond_h4_ex_covid(self):
        """The four-quarter cap on the published path rests on this."""
        rows = _load("unemployment_satellite_validation.json")["ex_covid_targets"]
        for row in rows:
            if row["horizon"] >= 5:
                assert row["ratio_vs_rw"]["svar"] > 1.0

    def test_all_rmse_values_are_finite_and_positive(self):
        report = _load("unemployment_satellite_validation.json")
        for label in ("all_targets", "ex_covid_targets"):
            for row in report[label]:
                for value in row["rmse"].values():
                    assert np.isfinite(value) and value > 0


# --------------------------------------------------------------------------
# 2. Regeneration smoke: the machinery re-run on real data, reduced scale
# --------------------------------------------------------------------------

class TestRegenerationSmoke:
    """Re-run the evaluation machinery on the packaged real data.

    Small configs (few origins / draws), so this stays in the fast suite:
    it cannot reproduce the headline numbers, but it fails if the
    evaluation code stops producing finite, well-formed, directionally
    sane output on the real dataset.
    """

    def test_rolling_evaluation_smoke_on_real_data(self):
        y = load_data().to_numpy(float)
        T, k = y.shape
        horizons = 4
        first_origin = T - horizons - 10  # 10 origins
        report = rolling_origin_evaluation(
            y, lags=4, horizons=horizons, first_origin=first_origin
        )
        assert report["origins"] == 10
        assert [row["horizon"] for row in report["horizons"]] == [1, 2, 3, 4]
        for row in report["horizons"]:
            for key in ("relative_rmse", "relative_rmse_vs_drift",
                        "relative_rmse_vs_ar1", "bvar_rmse",
                        "random_walk_rmse", "drift_rmse", "ar1_rmse"):
                vals = np.asarray(row[key])
                assert vals.shape == (k,)
                assert np.isfinite(vals).all(), f"non-finite {key}"
            assert (np.asarray(row["bvar_rmse"]) > 0).all()
            # Directional sanity: a fitted AR(1) must not be wildly worse
            # than no-change on any variable.
            ar1 = np.asarray(row["ar1_rmse"])
            rw = np.asarray(row["random_walk_rmse"])
            assert (ar1 < 5 * rw).all(), "AR1 benchmark exploded vs RW"
            # And the BVAR must be in the same league as the benchmarks.
            assert (np.asarray(row["relative_rmse"]) < 5).all()

    def test_coverage_evaluation_smoke_on_real_data(self):
        sys.path.insert(0, str(SCRIPTS))
        try:
            from run_coverage_evaluation import run_coverage
        finally:
            sys.path.pop(0)
        y = load_data().to_numpy(float)
        T, k = y.shape
        horizons = 4
        first_origin = T - horizons - 8  # 8 origins
        hits, n = run_coverage(
            y, lags=4, horizons=horizons, first_origin=first_origin,
            n_draws=25, n_paths=2, seed=0,
        )
        assert (n == 8).all()
        for lv in (68, 90):
            assert hits[lv].shape == (horizons, k)
            assert np.isfinite(hits[lv]).all()
            assert ((hits[lv] >= 0) & (hits[lv] <= n[:, None])).all()
        # 90% bands contain 68% bands, so hits can never be fewer.
        assert (hits[90] >= hits[68]).all()
        # Bands built from posterior draws must not be degenerate: across
        # the whole grid the 90% band has to catch a decent share.
        assert hits[90].sum() / (n.sum() * k) > 0.3

    def test_coverage_evaluation_accepts_covid_dummies(self):
        """The published forecast is estimated WITH the six Covid dummies, so
        the coverage that calibrates its bands has to be measurable under that
        specification too. Pins that the dummy path runs and gives coverage in
        [0, 1] on the real data."""
        sys.path.insert(0, str(SCRIPTS))
        try:
            from run_coverage_evaluation import run_coverage
        finally:
            sys.path.pop(0)
        df = load_data()
        y = df.to_numpy(float)
        T, k = y.shape
        horizons = 3
        quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
        covid = np.column_stack(
            [(df.index == q).astype(float) for q in quarters]
        )
        hits, n = run_coverage(
            y, lags=4, horizons=horizons, first_origin=T - horizons - 6,
            n_draws=20, n_paths=2, seed=0, dummies=covid,
        )
        assert (n == 6).all()
        for lv in (68, 90):
            assert ((hits[lv] >= 0) & (hits[lv] <= n[:, None])).all()
        assert (hits[90] >= hits[68]).all()


class TestBandScaleFactorDerivation:
    """The published fan charts are rescaled by factors derived from the
    coverage artifact. The derivation must be reproducible from the numbers
    in that same artifact, not taken on trust."""

    def _factors(self):
        sys.path.insert(0, str(SCRIPTS))
        try:
            from run_coverage_evaluation import band_scale_factors
        finally:
            sys.path.pop(0)
        return band_scale_factors

    def test_factor_is_the_gaussian_half_width_ratio(self):
        band_scale_factors = self._factors()
        # Coverage exactly at the nominal level needs no correction.
        exact = {"68": {"h1": {"uk_gdp": 0.68}}, "90": {"h1": {"uk_gdp": 0.90}}}
        out = band_scale_factors(exact)
        assert abs(out["68"]["h1"]["uk_gdp"] - 1.0) < 1e-3
        assert abs(out["90"]["h1"]["uk_gdp"] - 1.0) < 1e-3
        # Under-covering widens (k > 1), over-covering narrows (k < 1).
        under = {"68": {"h1": {"v": 0.40}}, "90": {"h1": {"v": 0.70}}}
        over = {"68": {"h1": {"v": 0.90}}, "90": {"h1": {"v": 0.98}}}
        assert band_scale_factors(under)["68"]["h1"]["v"] > 1.0
        assert band_scale_factors(over)["68"]["h1"]["v"] < 1.0
        # Degenerate coverage yields no factor rather than an invented one.
        degenerate = {"68": {"h1": {"v": 0.0}}, "90": {"h1": {"v": 1.0}}}
        assert band_scale_factors(degenerate)["68"]["h1"]["v"] is None
        assert band_scale_factors(degenerate)["90"]["h1"]["v"] is None

    def test_committed_factors_match_the_committed_coverage(self):
        band_scale_factors = self._factors()
        art = _load("coverage_evaluation.json")
        if "band_scale_factors" not in art:
            pytest.skip("artifact predates the published factor block")
        for block in ("headline", "production_spec"):
            source = (art["coverage"] if block == "headline"
                      else art["production_spec"]["coverage"])
            expected = band_scale_factors(source)
            assert art["band_scale_factors"][block] == expected, (
                f"{block} factors are not the ones implied by the coverage "
                "recorded in the same file")

    def test_published_spec_needs_wider_gdp_bands_than_the_headline_spec(self):
        """The finding this block exists to record.

        Calibration factors applied to the site's fan charts were derived from
        the headline (no-Covid-dummy) coverage. Dummying out the pandemic
        shrinks Sigma, so the published GDP bands are narrower and cover less
        than the ones that were measured: the correct factor is LARGER at
        every horizon. Using the headline factor therefore over-narrows the
        published GDP fan.
        """
        art = _load("coverage_evaluation.json")
        if "band_scale_factors" not in art:
            pytest.skip("artifact predates the published factor block")
        head = art["band_scale_factors"]["headline"]["68"]
        prod = art["band_scale_factors"]["production_spec"]["68"]
        bigger = sum(prod[f"h{h}"]["uk_gdp"] > head[f"h{h}"]["uk_gdp"]
                     for h in range(1, 9))
        assert bigger >= 6, (
            "the published specification no longer needs wider UK GDP bands "
            "than the headline specification; the recorded reading is stale")

    def test_factor_precision_is_not_mistaken_for_accuracy(self):
        """Coverage is measured on 49 OVERLAPPING origins, so a per-(variable,
        horizon) coverage carries at least the binomial standard error the
        artifact records -- about 6.7pp at the 68% level. Factors quoted to
        four decimals are far more precise than the measurement behind them,
        so the standard error has to travel with them."""
        art = _load("coverage_evaluation.json")
        se = art["coverage_standard_error"]
        assert se["68"] > 0.05 and se["90"] > 0.03
        emp = art["coverage"]["68"]["h1"]["uk_gdp"]
        # the plausible range of the factor implied by +-1 SE on coverage
        band_scale_factors = self._factors()
        lo = band_scale_factors(
            {"68": {"h": {"v": emp - se["68"]}}}, levels=(68,))["68"]["h"]["v"]
        hi = band_scale_factors(
            {"68": {"h": {"v": emp + se["68"]}}}, levels=(68,))["68"]["h"]["v"]
        assert hi < lo                      # more coverage -> smaller factor
        assert lo - hi > 0.05, (
            "a one-standard-error move in coverage should shift the factor "
            "far more than its quoted fourth decimal")


# --------------------------------------------------------------------------
# 3. Staleness guard: evaluation code vs committed artifacts
# --------------------------------------------------------------------------

class TestArtifactStaleness:
    """Fail when evaluation code changes without regenerating the artifact.

    Each runner writes ``code_version`` -- a SHA-256 over the sources its
    numbers depend on (see ``boe_var.evaluation.evaluation_code_version``).
    If one of these tests fails, re-run the matching script (each takes a
    few seconds) and commit the refreshed JSON:

        uv run python scripts/run_rolling_evaluation.py
        uv run python scripts/run_coverage_evaluation.py
        uv run python scripts/run_satellite_validation.py
    """

    def test_rolling_evaluation_is_not_stale(self):
        recorded = _load("rolling_evaluation.json")["code_version"]
        current = evaluation_code_version(
            extra_paths=[SCRIPTS / "run_rolling_evaluation.py"]
        )
        assert recorded == current, (
            "evaluation code changed since results/rolling_evaluation.json "
            "was generated; re-run scripts/run_rolling_evaluation.py"
        )

    def test_coverage_evaluation_is_not_stale(self):
        recorded = _load("coverage_evaluation.json")["code_version"]
        current = evaluation_code_version(
            extra_paths=[SCRIPTS / "run_coverage_evaluation.py"]
        )
        assert recorded == current, (
            "evaluation code changed since results/coverage_evaluation.json "
            "was generated; re-run scripts/run_coverage_evaluation.py"
        )

    def test_satellite_validation_is_not_stale(self):
        from boe_var import unemployment_satellite as satellite

        recorded = _load("unemployment_satellite_validation.json")["code_version"]
        current = evaluation_code_version(
            extra_paths=[satellite.__file__, SCRIPTS / "run_satellite_validation.py"]
        )
        assert recorded == current, (
            "evaluation code changed since "
            "results/unemployment_satellite_validation.json was generated; "
            "re-run scripts/run_satellite_validation.py"
        )

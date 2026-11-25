"""Impulse-response magnitudes against Brignone & Piffer (2025) Figures 2-3
(issue #11).

Three layers:

1. **The paper's medians as read from the PDF**
   (``data/external/brignone_piffer_2025_fig2_fig3_irf_medians.csv``). The
   file must be complete, must satisfy the paper's own Table 2 on impact
   (zero where a zero restriction applies, the restricted sign elsewhere),
   and, when PyMuPDF is installed, must be what
   ``scripts/extract_paper_irf_medians.py`` produces from the committed PDF.
2. **This replication's pinned impulse responses**
   (``results/irf_reference.json``): well-formed, not stale, and with the
   headline magnitudes the README quotes.
3. **The comparison.** The two are not expected to agree closely (proxy
   world series, a longer sample, a later data vintage), so the gates are
   the measured agreement with a margin: they fail if a code change moves
   the replication materially away from the paper, not if it fails to match.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from boe_var.identification import (
    SIGN_RESTRICTIONS,
    VARIABLES,
    ZERO_RESTRICTIONS,
)

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "external" / "brignone_piffer_2025_fig2_fig3_irf_medians.csv"
REFERENCE = REPO / "results" / "irf_reference.json"
SCRIPTS = REPO / "scripts"

SHOCKS = ["world_demand", "world_supply", "world_energy",
          "uk_demand", "uk_supply", "uk_monpol"]


def _script(name):
    sys.path.insert(0, str(SCRIPTS))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="module")
def paper():
    """{(shock, variable): {"median": array(21), "resolution": float}}."""
    out = {}
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            cell = out.setdefault(
                (r["shock"], r["variable"]),
                {"median": np.full(21, np.nan),
                 "resolution": float(r["resolution"]), "unit": r["unit"]})
            cell["median"][int(r["horizon"])] = float(r["median"])
    return out


@pytest.fixture(scope="module")
def reference():
    return json.loads(REFERENCE.read_text())


# --------------------------------------------------------------------------
# 1. The paper's medians
# --------------------------------------------------------------------------

class TestPaperMedians:
    def test_every_panel_and_quarter_is_present(self, paper):
        assert set(paper) == {(s, v) for s in SHOCKS for v in VARIABLES}
        for cell in paper.values():
            assert np.isfinite(cell["median"]).all()

    def test_units(self, paper):
        for (_, var), cell in paper.items():
            assert cell["unit"] == (
                "basis points" if var == "bank_rate" else "percent")

    def test_reading_resolution_is_small_against_the_responses(self, paper):
        """Half a coordinate-grid step, in each panel's units, against the
        panel's largest median response: 0.8% for the median panel, over 3%
        for three panels whose response stays near zero, and under 6%
        everywhere."""
        ratios = {k: c["resolution"] / np.abs(c["median"]).max()
                  for k, c in paper.items()}
        assert np.median(list(ratios.values())) < 0.01
        coarse = {k for k, r in ratios.items() if r > 0.03}
        assert coarse == {("uk_demand", "world_cpi"),
                          ("uk_supply", "bank_rate"),
                          ("uk_supply", "oil_price")}
        assert max(ratios.values()) < 0.06

    def test_zero_restrictions_of_table_2_hold_on_impact(self, paper):
        """An independent check on the reading: where Table 2 imposes a
        zero, every draw is zero on impact, so the median must be too."""
        for shock in SHOCKS:
            for i in ZERO_RESTRICTIONS[shock]:
                cell = paper[(shock, VARIABLES[i])]
                assert abs(cell["median"][0]) <= cell["resolution"], (
                    shock, VARIABLES[i])

    def test_sign_restrictions_of_table_2_hold_on_impact(self, paper):
        """Where Table 2 imposes a sign, every draw has it on impact, so the
        median must too. This also ties the restriction table coded in
        ``identification.py`` to the paper's own figures."""
        for shock in SHOCKS:
            for i, sign in SIGN_RESTRICTIONS[shock]:
                cell = paper[(shock, VARIABLES[i])]
                assert sign * cell["median"][0] > cell["resolution"], (
                    shock, VARIABLES[i])

    def test_prose_claims_of_section_3_1(self, paper):
        """The paper's text (p. 11) describes its own figures; the numbers
        read from them must agree with it."""
        # "the oil price variable, which is unrestricted, increases in
        # response to the expansionary supply shock"
        assert (paper[("world_supply", "oil_price")]["median"] > 0).all()
        # world demand: "a large, positive, but relatively short-lived,
        # impact on UK real GDP"
        gdp = paper[("world_demand", "uk_gdp")]["median"]
        assert gdp[:4].min() > 0.15 and abs(gdp[12:]).max() < 0.03
        # UK supply: "a persistent increase of the level of UK GDP"
        assert paper[("uk_supply", "uk_gdp")]["median"][20] > 0.25
        # monetary policy: "a negative response of CPI energy and World CPI"
        assert (paper[("uk_monpol", "cpi_energy")]["median"] < 0).all()
        assert (paper[("uk_monpol", "world_cpi")]["median"][1:] < 0).all()

    def test_monetary_policy_shock_headline_values(self, paper):
        """One standard deviation: Bank Rate +8bp on impact, peak 12bp;
        UK GDP about -0.1%; UK CPI -0.4% after five years."""
        rate = paper[("uk_monpol", "bank_rate")]["median"]
        assert rate[0] == pytest.approx(8.07, abs=0.1)
        assert rate.max() == pytest.approx(12.0, abs=0.5)
        assert paper[("uk_monpol", "uk_gdp")]["median"][0] == pytest.approx(
            -0.116, abs=0.005)
        assert paper[("uk_monpol", "cpisa")]["median"][20] == pytest.approx(
            -0.42, abs=0.02)

    def test_csv_is_what_the_script_reads_from_the_pdf(self, tmp_path):
        """Re-read the PDF when PyMuPDF is available (it is not a project
        dependency: AGPL)."""
        pytest.importorskip("pymupdf")
        mod = _script("extract_paper_irf_medians")
        out = tmp_path / "medians.csv"
        mod.write(mod.extract(), str(out))
        assert out.read_text() == CSV.read_text()

    def test_scale_check_rejects_a_misread_label(self):
        """The four checks in the extraction script are what stand between a
        transcription slip and a wrong number. Exercise them without the
        PDF: three labels drawn 9 pt apart, zero line at y = 100."""
        mod = _script("extract_paper_irf_medians")
        rows = [(109.42, 1), (100.42, 1), (91.42, 1)]   # bottom to top
        per_unit, misfit = mod._scale(rows, ["-1", "0", "1"], 100.0, "t")
        assert per_unit == pytest.approx(9.0)
        assert misfit < 1e-9
        with pytest.raises(mod.ExtractionError, match="evenly spaced"):
            mod._scale(rows, ["-1", "0", "2"], 100.0, "t")       # digit slip
        with pytest.raises(mod.ExtractionError, match="digits"):
            mod._scale(rows, ["-10", "0", "10"], 100.0, "t")     # lost a digit
        with pytest.raises(mod.ExtractionError, match="zero line"):
            mod._scale(rows, ["0", "1", "2"], 100.0, "t")        # lost a sign
        with pytest.raises(mod.ExtractionError, match="label rows"):
            mod._scale(rows, ["0", "1"], 100.0, "t")


# --------------------------------------------------------------------------
# 2. This replication's pinned impulse responses
# --------------------------------------------------------------------------

class TestReference:
    """results/irf_reference.json -- the published run's impulse responses.

    Two operating systems reproduced the file to 3e-6 (the manual
    ``irf-reference`` job in ci.yml); the tolerance below absorbs that."""

    ATOL = 2e-3

    def test_run_is_the_published_replication(self, reference):
        assert reference["draws"] == 10_000
        assert reference["seed"] == 20260812
        assert reference["sample"] == {"start": "1992Q1", "end": "2025Q1"}
        assert reference["hyperparameters"] == {
            "lam": 0.2, "mu": 1.0, "theta": 1.0}
        # the counts results/summary.md reports for the same run
        summary = (REPO / "results" / "summary.md").read_text()
        assert f"accepted identified draws: {reference['accepted']} " in summary
        assert f"(ESS): {reference['ess']}." in summary

    def test_is_not_stale(self, reference):
        mod = _script("run_irf_reference")
        assert reference["code_version"] == mod.code_version(), (
            "impulse-response code, data or the paper CSV changed without "
            "regenerating results/irf_reference.json: dispatch the CI "
            "workflow (job irf-reference), or run "
            "scripts/run_irf_reference.py, and commit the result")

    def test_shape_and_ordering_of_the_sets(self, reference):
        assert reference["shocks"] == SHOCKS
        assert reference["variables"] == VARIABLES
        for shock in SHOCKS:
            for var in VARIABLES:
                cell = reference["irf"][shock][var]
                arr = {k: np.asarray(v) for k, v in cell.items()}
                assert all(a.shape == (21,) for a in arr.values())
                assert (arr["lo90"] <= arr["lo68"] + 1e-9).all()
                assert (arr["lo68"] <= arr["median"] + 1e-9).all()
                assert (arr["median"] <= arr["hi68"] + 1e-9).all()
                assert (arr["hi68"] <= arr["hi90"] + 1e-9).all()

    def test_table_2_holds_on_impact_for_the_whole_90_set(self, reference):
        """Restrictions hold on every draw, so on impact the 90% set is on
        the restricted side of zero, and is zero where a zero is imposed."""
        for shock in SHOCKS:
            for i in ZERO_RESTRICTIONS[shock]:
                cell = reference["irf"][shock][VARIABLES[i]]
                assert abs(cell["lo90"][0]) < 1e-6 > abs(cell["hi90"][0])
            for i, sign in SIGN_RESTRICTIONS[shock]:
                cell = reference["irf"][shock][VARIABLES[i]]
                assert sign * cell["lo90"][0] > 0 < sign * cell["hi90"][0]

    def test_monetary_policy_shock_magnitudes(self, reference):
        """One standard deviation, medians: Bank Rate +9.1bp on impact and
        still +5.0bp after five years; UK GDP -0.14% on impact and -0.17%
        after five years; UK CPI -0.09% on impact and -0.20% after five
        years. The README quotes these."""
        mp = reference["irf"]["uk_monpol"]
        pins = {("bank_rate", 0): 9.09, ("bank_rate", 20): 4.97,
                ("uk_gdp", 0): -0.140, ("uk_gdp", 20): -0.170,
                ("cpisa", 0): -0.094, ("cpisa", 20): -0.200}
        for (var, h), value in pins.items():
            tol = 0.02 if var == "bank_rate" else self.ATOL
            assert mp[var]["median"][h] == pytest.approx(value, abs=tol), (
                var, h)

    def test_reduced_scale_run_is_well_formed(self):
        """The runner end to end on the packaged data, at 300 draws."""
        mod = _script("run_irf_reference")
        report = mod.build_report(n_draws=300)
        assert report["accepted"] >= 5
        cmp_ = report["paper_comparison"]
        assert cmp_["points"] == 48 * 21
        assert 0.0 <= cmp_["share_inside_68"] <= cmp_["share_inside_90"] <= 1.0
        assert len(cmp_["impact_ratio_sign_restricted"]["by_response"]) == 23
        text = mod.markdown(report)
        assert text.count("\n| ") >= 48
        assert mod.max_difference(report, report) == 0.0
        other = json.loads(json.dumps(report))
        other["irf"]["uk_monpol"]["uk_gdp"]["median"][3] += 0.01
        assert mod.max_difference(report, other) == pytest.approx(0.01)
        other["accepted"] += 1
        assert mod.max_difference(report, other) == float("inf")


# --------------------------------------------------------------------------
# 3. This replication against the paper
# --------------------------------------------------------------------------

class TestAgainstPaper:
    """Floors on the measured agreement (results/irf_vs_paper.md). Each
    bound is the committed value with a margin, so the test fails when a
    change moves the replication away from the paper, and the committed
    value itself is pinned so the report cannot drift unnoticed."""

    def test_comparison_block_matches_the_two_committed_sources(
            self, reference, paper):
        """Recompute the summary shares from the CSV and the pinned
        responses, independently of the script."""
        inside68 = inside90 = agree = compared = 0
        for (shock, var), cell in paper.items():
            ours = reference["irf"][shock][var]
            pm = cell["median"]
            inside68 += ((np.asarray(ours["lo68"]) <= pm)
                         & (pm <= np.asarray(ours["hi68"]))).sum()
            inside90 += ((np.asarray(ours["lo90"]) <= pm)
                         & (pm <= np.asarray(ours["hi90"]))).sum()
            clear = np.abs(pm) > 3 * cell["resolution"]
            compared += clear.sum()
            agree += (np.sign(pm) == np.sign(ours["median"]))[clear].sum()
        cmp_ = reference["paper_comparison"]
        assert cmp_["share_inside_68"] == pytest.approx(inside68 / 1008, abs=1e-4)
        assert cmp_["share_inside_90"] == pytest.approx(inside90 / 1008, abs=1e-4)
        assert cmp_["sign_points_compared"] == compared
        assert cmp_["share_sign_agrees"] == pytest.approx(agree / compared,
                                                          abs=1e-4)

    def test_paper_median_is_mostly_inside_the_credible_sets(self, reference):
        cmp_ = reference["paper_comparison"]
        assert cmp_["share_inside_90"] == pytest.approx(0.929, abs=0.005)
        assert cmp_["share_inside_68"] == pytest.approx(0.765, abs=0.005)
        assert cmp_["share_inside_90"] > 0.90
        assert cmp_["share_inside_68"] > 0.70
        assert cmp_["panels_fully_inside_90"] == 41
        assert cmp_["panels_fully_outside_90"] == 0

    def test_medians_mostly_share_a_sign(self, reference):
        cmp_ = reference["paper_comparison"]
        assert cmp_["share_sign_agrees"] == pytest.approx(0.840, abs=0.005)
        assert cmp_["share_sign_agrees"] > 0.80

    def test_impact_magnitudes_of_sign_restricted_responses(self, reference):
        """On impact, where Table 2 fixes the sign, this replication's
        median is 0.94 to 1.57 times the paper's (median 1.21) for 22 of 23
        responses. World supply -> UK CPI is the exception at 3.2 (-0.13%
        against the paper's -0.04%)."""
        imp = reference["paper_comparison"]["impact_ratio_sign_restricted"]
        by = dict(imp["by_response"])
        outlier = by.pop("world_supply:cpisa")
        assert len(by) == 22
        assert min(by.values()) > 0.8
        assert max(by.values()) < 1.8
        assert imp["median"] == pytest.approx(1.21, abs=0.03)
        assert outlier == pytest.approx(3.24, abs=0.1)

    def test_known_differences_from_the_paper_are_still_the_recorded_ones(
            self, reference):
        """The panels where the paper's median leaves this replication's 90%
        set at some quarter. Recorded so that a change which adds a panel,
        or removes one, is noticed and the README updated."""
        outside = {(c["shock"], c["variable"])
                   for c in reference["paper_comparison"]["cells"]
                   if c["inside_90"] < 21}
        assert outside == {
            ("world_demand", "world_cpi"), ("world_supply", "world_cpi"),
            ("uk_demand", "oil_price"), ("uk_supply", "world_gdp"),
            ("uk_supply", "world_cpi"), ("uk_monpol", "world_cpi"),
            ("uk_monpol", "cpisa"),
        }

    def test_monetary_policy_persistence_gap(self, reference, paper):
        """The clearest economic difference. In the paper a monetary-policy
        shock's Bank Rate response is back to 1bp after five years; here it
        is still 5bp. The paper's UK CPI response is about twice as large
        at five years (-0.42% against -0.20%)."""
        ours = reference["irf"]["uk_monpol"]
        theirs_rate = paper[("uk_monpol", "bank_rate")]["median"]
        assert theirs_rate[20] / theirs_rate[0] < 0.15
        assert ours["bank_rate"]["median"][20] / ours["bank_rate"]["median"][0] > 0.4
        ratio = (ours["cpisa"]["median"][20]
                 / paper[("uk_monpol", "cpisa")]["median"][20])
        assert ratio == pytest.approx(0.47, abs=0.05)

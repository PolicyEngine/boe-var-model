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

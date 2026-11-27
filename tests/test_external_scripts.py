"""Hermetic tests of the scoring and seasonal-check scripts' arithmetic and
of the release dating in the committed external records.

``tests/test_external_records.py`` checks that ``scripts/score_2026q3.py``
and ``scripts/compare_ecb_seasonal.py`` reproduce their committed reports.
That holds the output still; it does not say the arithmetic is right. These
tests work the outturns out by hand, run the scoring on a small synthetic
record, and check the dates that make the 2026Q3 score an out-of-sample one:
each stored forecast was archived before the release it is scored against,
and its data ended before the quarter scored.
"""

import csv
import importlib.util
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data" / "external"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(name):
    with open(EXT / name) as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------ score_2026q3

def test_outturns_are_the_hand_computed_rates():
    """From data/external/uk_outturns_2026q3.csv: GDP 749,999 against
    735,710; CPI 142.9, 143.6, 144.1 against 139.0, 139.3, 139.3;
    unemployed 1,684 thousand, employed 34,625 thousand."""
    o = _load("score_2026q3").outturns()
    assert o["gdp"] == pytest.approx(100 * (749999 / 735710 - 1))
    assert o["gdp"] == pytest.approx(1.9422, abs=5e-5)
    assert o["cpi"] == pytest.approx(100 * (430.6 / 417.6 - 1))
    assert o["cpi"] == pytest.approx(3.1130, abs=5e-5)
    assert o["unemployment"] == pytest.approx(100 * 1684 / 36309)
    assert o["unemployment"] == pytest.approx(4.6380, abs=5e-5)
    assert o["unemployment_published"] == 4.6
    assert o["bank_rate"] == 3.75


def test_outturns_on_a_synthetic_record(tmp_path, monkeypatch):
    """CPI is the ratio of three-month means, not the mean of three ratios."""
    mod = _load("score_2026q3")
    lines = ["series,code,source_dataset,release_date,period,value,unit",
             "uk_gdp,ABMI,x,2026-11-12,2025Q3,200,x",
             "uk_gdp,ABMI,x,2026-11-12,2026Q3,210,x"]
    for month, last, now in ((7, 100.0, 104.0), (8, 100.0, 104.0),
                             (9, 130.0, 131.0)):
        lines += [f"uk_cpi,D7BT,x,2026-11-18,2025-{month:02d},{last},x",
                  f"uk_cpi,D7BT,x,2026-11-18,2026-{month:02d},{now},x"]
    lines += ["uk_unemployed,MGSC,x,2026-11-17,2026Q3,50,x",
              "uk_employed,MGRZ,x,2026-11-17,2026Q3,950,x",
              "uk_unemployment_rate,MGSX,x,2026-11-17,2026Q3,5.0,x",
              "bank_rate,IUQABEDR,x,retrieved 2026-11-19,2026Q3,4.0,x"]
    (tmp_path / "uk_outturns_2026q3.csv").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(mod, "EXT", tmp_path)
    o = mod.outturns()
    assert o["gdp"] == pytest.approx(5.0)
    assert o["cpi"] == pytest.approx(100 * (339.0 / 330.0 - 1))   # 2.727
    assert o["cpi"] != pytest.approx((4.0 + 4.0 + 100 / 130) / 3, abs=0.05)
    assert o["unemployment"] == pytest.approx(5.0)
    assert o["bank_rate"] == 4.0


@pytest.mark.parametrize("x, expected", [
    (2.0, "inside 68%"), (1.0, "inside 68%"), (3.0, "inside 68%"),
    (0.99, "outside 68%, inside 90%"), (0.5, "outside 68%, inside 90%"),
    (3.5, "outside 68%, inside 90%"), (0.49, "outside 90%"),
    (3.51, "outside 90%"),
])
def test_where_counts_band_edges_as_inside(x, expected):
    band = {"lo68": 1.0, "hi68": 3.0, "lo90": 0.5, "hi90": 3.5}
    assert _load("score_2026q3").where(x, band) == expected


def test_log_basis_forecasts_are_scored_against_the_log_outturn(tmp_path,
                                                                capsys):
    """The 29 Jul round is on the log basis. Its 'outturn, same basis' cell
    must be 100*ln(1 + rate/100), below the simple rate."""
    mod = _load("score_2026q3")
    mod.OUT = tmp_path / "score.md"
    mod.main()
    capsys.readouterr()
    o = mod.outturns()
    table = [ln for ln in mod.OUT.read_text().splitlines()
             if ln.startswith("| 2026-")]
    assert len(table) == 4
    for line in table:
        cells = [c.strip() for c in line.strip("|").split("|")]
        rate = o[cells[1].lower()]
        expected = 100 * math.log1p(rate / 100) if cells[2] == "log" else rate
        assert cells[2] == ("log" if cells[0].startswith("2026-07-29")
                            else "simple")
        assert float(cells[6]) == pytest.approx(expected, abs=0.005)
        assert float(cells[7]) == pytest.approx(expected - float(cells[3]),
                                                abs=0.011)


# ---------------------------------------------------------- release dating

# ONS publishes at 07:00 UK time. In November that is 07:00 UTC.
ONS_RELEASE_HOUR_UTC = 7
SCORED = {"gdp": "ABMI", "cpi": "D7BT"}


def _release(row) -> datetime:
    d = datetime.strptime(row["release_date"], "%Y-%m-%d")
    return d.replace(hour=ONS_RELEASE_HOUR_UTC, tzinfo=timezone.utc)


def _period_end(period: str) -> pd.Timestamp:
    freq = "Q" if "Q" in period else "M"
    return pd.Period(period, freq=freq).end_time


def test_every_ons_outturn_was_released_after_its_period_ended():
    rows = [r for r in _rows("uk_outturns_2026q3.csv")
            if r["source_dataset"].startswith("ONS")]
    assert len(rows) == 11
    for r in rows:
        released = pd.Timestamp(r["release_date"])
        assert released > _period_end(r["period"]), r
        # A first estimate, not a later vintage: within two months of the
        # end of 2026Q3.
        assert released <= pd.Timestamp("2026-11-30"), r


def test_rows_of_one_release_carry_one_release_date():
    dates = {}
    for r in _rows("uk_outturns_2026q3.csv"):
        dates.setdefault(r["source_dataset"], set()).add(r["release_date"])
    assert all(len(v) == 1 for v in dates.values()), dates
    assert dates["ONS PN2 (GDP first quarterly estimate)"] == {"2026-11-12"}
    assert dates["ONS MM23"] == {"2026-11-18"}
    assert dates["ONS LMS"] == {"2026-11-17"}


def test_each_stored_forecast_was_archived_before_the_outturn_it_is_scored_on():
    """The 12 Nov round is the close one: archived at 05:13 UTC on the day
    ONS published the GDP first estimate at 07:00."""
    outturns = _rows("uk_outturns_2026q3.csv")
    release = {code: min(_release(r) for r in outturns if r["code"] == code)
               for code in SCORED.values()}
    forecasts = _rows("svar_stored_forecasts_2026q3.csv")
    assert len(forecasts) == 4
    for f in forecasts:
        archived = datetime.strptime(
            f["archived_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        assert archived < release[SCORED[f["variable"]]], f
        assert f["round"] == f["archived_utc"][:10]
    gdp_12nov = next(f for f in forecasts
                     if f["round"] == "2026-11-12" and f["variable"] == "gdp")
    assert gdp_12nov["archived_utc"] == "2026-11-12T05:13:50Z"


def test_each_stored_forecast_is_out_of_sample_for_the_quarter_scored():
    for f in _rows("svar_stored_forecasts_2026q3.csv"):
        assert f["quarter"] == "2026Q3"
        assert pd.Period(f["data_edge"], "Q") < pd.Period(f["quarter"], "Q")
        # The model could not have been run before its last data quarter ended.
        assert pd.Timestamp(f["archived_utc"][:10]) > pd.Period(
            f["data_edge"], "Q").end_time


def test_the_committed_dataset_ends_at_the_latest_stored_data_edge():
    """The 12 Nov round's data edge (2026Q2) is the committed dataset's last
    quarter, so the score mixes no later vintage into the forecast."""
    d = pd.read_csv(ROOT / "data" / "boe_var_data.csv")
    edges = {f["data_edge"] for f in _rows("svar_stored_forecasts_2026q3.csv")}
    assert d["quarter"].iloc[-1] == max(edges) == "2026Q2"


# ---------------------------------------------------- compare_ecb_seasonal

def test_quarterly_keeps_only_quarters_with_three_months():
    mod = _load("compare_ecb_seasonal")
    months = pd.period_range("2020-02", "2020-10", freq="M")
    s = pd.Series(np.arange(2.0, 11.0), index=months)   # value = month number
    out = mod.quarterly(s)
    # 2020Q1 has two months and 2020Q4 one: both dropped.
    assert out.to_dict() == {pd.Period("2020Q2", "Q"): 5.0,
                             pd.Period("2020Q3", "Q"): 8.0}


def test_f_test_separates_a_seasonal_series_from_an_unseasonal_one():
    mod = _load("compare_ecb_seasonal")
    idx = pd.period_range("2000Q1", "2019Q4", freq="Q")
    noise = np.random.default_rng(0).normal(0.0, 0.1, len(idx))
    flat = pd.Series(0.5 + noise, index=idx)
    seasonal = flat + np.array([-0.2, 0.4, -0.3, 0.1])[idx.quarter - 1]
    assert mod.f_test(seasonal, "2000Q1", "2019Q4").pvalue < 1e-6
    assert mod.f_test(flat, "2000Q1", "2019Q4").pvalue > 0.10
    # The window is honoured: a seasonal confined to 2010-19 is not seen
    # in 2000-09.
    late = flat + np.where(idx.year >= 2010,
                           np.array([-0.2, 0.4, -0.3, 0.1])[idx.quarter - 1], 0.0)
    assert mod.f_test(late, "2000Q1", "2009Q4").pvalue > 0.10
    assert mod.f_test(late, "2010Q1", "2019Q4").pvalue < 1e-6


def test_ecb_file_has_whole_months_and_both_indices():
    m = pd.read_csv(EXT / "ecb_hicp_euro_area_monthly.csv")
    idx = pd.PeriodIndex(m["month"], freq="M")
    assert list(m.columns) == ["month", "hicp_sa_ecb", "hicp_nsa_eurostat"]
    assert str(idx[0]) == "1997-01"
    assert (np.diff(idx.asi8) == 1).all()          # no gaps, no repeats
    assert m[["hicp_sa_ecb", "hicp_nsa_eurostat"]].notna().all().all()

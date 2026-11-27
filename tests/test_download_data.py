"""Hermetic tests of ``scripts/download_data.py`` (no network, no data/raw).

The script downloads eighteen raw files from ONS, the Bank of England and
FRED and turns them into ``data/boe_var_data.csv``. Before this file only
``seasonal_adjust_tv`` was tested, on the committed output. Here:

* the three parsers read small files written in each publisher's layout;
* every transform is checked against a number worked out by hand;
* ``build()`` runs end to end on synthetic raw files in which each series
  grows at a constant, known rate, so every column of the result has a
  closed form: the transforms, the German backcasts, the US CPI splice, the
  Japanese CPI extension, the era trade weights and their renormalisation
  when a country has no data, and the last quarter of the sample;
* ``download_all`` runs against a stub of ``requests.get``;
* ``main()`` writes its four files under ``tmp_path``.

Every test runs with sockets disabled, ``time.sleep`` stubbed, and the
script's ``ROOT`` / ``RAW`` / ``OUT`` pointed at ``tmp_path``, so nothing
here can reach a publisher or overwrite a committed file.

Five tests record behaviour that is questionable rather than assert that it
is right; each says so in its docstring (issue #41).
"""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Quarterly log growth of each synthetic series (monthly series: per month).
G_UK_GDP = 0.005
G_US_GDP, G_EA_GDP, G_DE_GDP, G_JP_GDP = 0.006, 0.004, 0.003, 0.002
G_CN_GDP_ANNUAL = 0.08
G_US_CPI_OECD, G_US_CPI_BLS_M = 0.006, 0.003
G_EA_CPI_M, G_DE_CPI_M = 0.0015, 0.002
G_JP_CPI, JP_ANNUAL_PCT = 0.001, 2.0
G_CN_CPI_M = 0.004
G_UK_CPI_M, G_E1_M, G_E2_M = 0.002, 0.003, 0.005
G_BRENT, G_ERI = 0.01, -0.001
USD_PER_GBP = 1.5

ERA1 = {"ea": 0.50, "us": 0.17, "jp": 0.04, "cn": 0.01}
ERA2 = {"ea": 0.48, "us": 0.16, "jp": 0.03, "cn": 0.05}
ERA3 = {"ea": 0.45, "us": 0.16, "jp": 0.02, "cn": 0.07}


# ------------------------------------------------------------------ fixtures

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail the test, loudly, if anything opens a socket."""
    def refuse(*args, **kwargs):
        raise AssertionError("a download_data test tried to open a socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def dd(tmp_path, monkeypatch):
    """The script as a module, with every path under tmp_path and no sleeps."""
    spec = importlib.util.spec_from_file_location(
        "download_data_under_test", ROOT / "scripts" / "download_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "RAW", tmp_path / "data" / "raw")
    monkeypatch.setattr(mod, "OUT", tmp_path / "data" / "boe_var_data.csv")
    mod.RAW.mkdir(parents=True)
    mod.slept = []
    monkeypatch.setattr(mod.time, "sleep", mod.slept.append)
    return mod


def _q(a: str, b: str) -> pd.PeriodIndex:
    return pd.period_range(a, b, freq="Q")


def _m(a: str, b: str) -> pd.PeriodIndex:
    return pd.period_range(a, b, freq="M")


def _growing(index, rate: float, start: float = 100.0) -> pd.Series:
    return pd.Series(start * np.exp(rate * np.arange(len(index))), index=index)


def _ons_text(cdid: str, monthly: pd.Series | None = None,
              quarterly: pd.Series | None = None) -> str:
    """An ONS generator CSV: metadata rows, then annual, quarterly and
    monthly rows in one file, every field quoted."""
    lines = ['"Title","Synthetic series"', f'"CDID","{cdid}"',
             '"Source dataset ID","TEST"', '"PreUnit",""', '"Unit","Index"',
             '"Release date","30-09-2026"', '"Next release","22 December 2026"',
             '"Important notes",']
    src = monthly if monthly is not None else quarterly
    for year in sorted({p.year for p in src.index}):
        lines.append(f'"{year}","999999"')          # annual rows: not wanted
    if quarterly is not None:
        for p, v in quarterly.items():
            lines.append(f'"{p.year} Q{p.quarter}","{v:.10f}"')
    if monthly is not None:
        for year in sorted({p.year for p in monthly.index}):
            for k in (1, 2, 3, 4):
                lines.append(f'"{year} Q{k}","888888"')  # quarterly: not wanted
        for p, v in monthly.items():
            lines.append(f'"{p.year} {p.strftime("%b").upper()}","{v:.10f}"')
    return "\n".join(lines) + "\n"


def _fred_text(sid: str, s: pd.Series) -> str:
    """A FRED graph CSV: observation_date at the start of each period."""
    rows = [f"observation_date,{sid}"]
    for p, v in s.items():
        rows.append(f"{p.start_time.strftime('%Y-%m-%d')},{v:.12f}")
    return "\n".join(rows) + "\n"


def _boe_text(bank_rate: pd.Series, eri: pd.Series, usd: pd.Series) -> str:
    """A BoE IADB CSV: quarter-end dates such as '31 Mar 1988'."""
    rows = ["DATE,IUQABEDR,XUQABK67,XUQAUSS"]
    for p in bank_rate.index:
        rows.append(f"{p.end_time.strftime('%d %b %Y')},{bank_rate[p]:.6f},"
                    f"{eri[p]:.10f},{usd[p]:.6f}")
    return "\n".join(rows) + "\n"


def raw_series() -> dict[str, pd.Series]:
    """The eighteen synthetic sources, with the real files' ragged edges.

    Starts and ends copy the 6 Nov 2026 vintage (data/README.md): Brent from
    1992Q1; EA19 GDP from 1995Q1 and Japanese GDP from 1994Q1, both to
    2026Q2; Chinese GDP annual to 2023; the OECD US CPI to 2025Q1; Japanese
    CPI to 2021Q2 with annual inflation to 2025; Chinese CPI from January
    1993 to April 2025. Euro-area HICP starts in January 1997 here, not
    December 1996 (see test_a_part_quarter_is_averaged...).
    """
    cn_years = pd.period_range("1988", "2023", freq="Y")
    jp_years = pd.period_range("1988", "2025", freq="Y")
    return {
        "ons_abmi": _growing(_q("1988Q1", "2026Q2"), G_UK_GDP, 400000.0),
        "ons_d7bt": _growing(_m("1988-01", "2026-09"), G_UK_CPI_M, 50.0),
        "ons_d7ch": _growing(_m("1988-01", "2026-09"), G_E1_M, 40.0),
        "ons_d7ec": _growing(_m("1988-01", "2026-09"), G_E2_M, 30.0),
        "boe_bank_rate": pd.Series(5.0 + 0.01 * np.arange(155),
                                   index=_q("1988Q1", "2026Q3")),
        "boe_eri": _growing(_q("1988Q1", "2026Q3"), G_ERI, 100.0),
        "boe_usd": pd.Series(USD_PER_GBP, index=_q("1988Q1", "2026Q3")),
        "fred_POILBREUSDQ": _growing(_q("1992Q1", "2026Q3"), G_BRENT, 20.0),
        "fred_GDPC1": _growing(_q("1988Q1", "2026Q3"), G_US_GDP, 9000.0),
        "fred_CLVMNACSCAB1GQEA19": _growing(_q("1995Q1", "2026Q2"), G_EA_GDP,
                                            2000000.0),
        "fred_CLVMNACSCAB1GQDE": _growing(_q("1988Q1", "2026Q2"), G_DE_GDP,
                                          500000.0),
        "fred_JPNRGDPEXP": _growing(_q("1994Q1", "2026Q2"), G_JP_GDP, 450000.0),
        "fred_RGDPNACNA666NRUG": _growing(cn_years, G_CN_GDP_ANNUAL, 2.0e6),
        "fred_CPALTT01USQ661S": _growing(_q("1988Q1", "2025Q1"), G_US_CPI_OECD,
                                         50.0),
        "fred_CPIAUCSL": _growing(_m("1988-01", "2026-09"), G_US_CPI_BLS_M,
                                  115.0),
        "fred_CP0000EZ19M086NEST": _growing(_m("1997-01", "2026-09"),
                                            G_EA_CPI_M, 70.0),
        "fred_DEUCPIALLMINMEI": _growing(_m("1988-01", "2026-09"), G_DE_CPI_M,
                                         60.0),
        "fred_CPALTT01JPQ661S": _growing(_q("1988Q1", "2021Q2"), G_JP_CPI,
                                         90.0),
        "fred_FPCPITOTLZGJPN": pd.Series(JP_ANNUAL_PCT, index=jp_years),
        "fred_CHNCPIALLMINMEI": _growing(_m("1993-01", "2025-04"), G_CN_CPI_M,
                                         40.0),
    }


def write_raw(raw: Path, **changes: pd.Series) -> None:
    """Write the eighteen raw files; ``changes`` replaces named sources."""
    s = raw_series() | changes
    files = {
        "ons_abmi.csv": _ons_text("ABMI", quarterly=s["ons_abmi"]),
        "ons_d7bt.csv": _ons_text("D7BT", monthly=s["ons_d7bt"]),
        "ons_d7ch.csv": _ons_text("D7CH", monthly=s["ons_d7ch"]),
        "ons_d7ec.csv": _ons_text("D7EC", monthly=s["ons_d7ec"]),
        "boe_rates_fx.csv": _boe_text(s["boe_bank_rate"], s["boe_eri"],
                                      s["boe_usd"]),
    }
    for name, series in s.items():
        if name.startswith("fred_"):
            files[f"{name}.csv"] = _fred_text(name.removeprefix("fred_"), series)
    for name, text in files.items():
        (raw / name).write_text(text)


@pytest.fixture
def built(dd):
    write_raw(dd.RAW)
    df, df_v1 = dd.build()
    return dd, df, df_v1


def _quarters_since(start: str, quarter: str) -> int:
    return (pd.Period(quarter, "Q") - pd.Period(start, "Q")).n


def _weighted(weights: dict, growth: dict) -> float:
    """Era weights renormalised over the countries in ``growth``."""
    total = sum(weights[c] for c in growth)
    return sum(weights[c] * g for c, g in growth.items()) / total


# ------------------------------------------------------------------- parsers

def test_read_ons_quarterly_keeps_only_quarter_rows(dd):
    q = pd.Series([1.5, 2.5, 3.5], index=_q("1990Q1", "1990Q3"))
    (dd.RAW / "x.csv").write_text(_ons_text("ABMI", quarterly=q))
    out = dd.read_ons("x.csv", "Q")
    assert list(out.index) == list(q.index)      # no metadata or annual rows
    assert out.index.freqstr.startswith("Q")
    assert out.tolist() == [1.5, 2.5, 3.5]


def test_read_ons_monthly_keeps_only_month_rows(dd):
    m = pd.Series(np.arange(1.0, 15.0), index=_m("1990-01", "1991-02"))
    (dd.RAW / "x.csv").write_text(_ons_text("D7BT", monthly=m))
    out = dd.read_ons("x.csv", "M")
    # The file also holds annual ("1990") and quarterly ("1990 Q1") rows with
    # the values 999999 and 888888; neither may leak into the monthly series.
    assert list(out.index) == list(m.index)
    assert out.tolist() == m.tolist()
    assert out.loc["1990-12"] == 12.0 and out.loc["1991-01"] == 13.0


def test_read_ons_skips_the_release_date_header_and_blank_values(dd):
    (dd.RAW / "x.csv").write_text(
        '"Title","T"\n"Release date","30-09-2026"\n"Next release","22 December 2026"\n'
        '"2026 Q1","10"\n"2026 Q2",""\n"2026 Q3","12"\n')
    out = dd.read_ons("x.csv", "Q")
    assert out.to_dict() == {pd.Period("2026Q1", "Q"): 10.0,
                             pd.Period("2026Q3", "Q"): 12.0}


def test_read_ons_raises_when_no_row_matches(dd):
    (dd.RAW / "x.csv").write_text('"Title","T"\n"1990","5"\n')
    with pytest.raises(ValueError, match="no Q rows parsed"):
        dd.read_ons("x.csv", "Q")


def test_read_boe_dates_quarter_ends_to_quarters(dd):
    (dd.RAW / "b.csv").write_text(
        "DATE,IUQABEDR,XUQABK67,XUQAUSS\n"
        "31 Mar 1988,8.6328,97.7071,1.7972\n"
        "30 Jun 1988,7.9652,100.936,1.839\n"
        "31 Dec 1988,12.1,98.0,1.8\n")
    out = dd.read_boe("b.csv")
    assert [str(p) for p in out.index] == ["1988Q1", "1988Q2", "1988Q4"]
    assert out.loc["1988Q1", "IUQABEDR"] == 8.6328
    assert out.loc["1988Q2", "XUQAUSS"] == 1.839
    assert list(out.columns) == ["IUQABEDR", "XUQABK67", "XUQAUSS"]


def test_read_fred_drops_missing_observations(dd):
    # FRED writes a missing observation as "." (or leaves it empty).
    (dd.RAW / "f.csv").write_text(
        "observation_date,X\n2020-01-01,1.0\n2020-02-01,.\n2020-03-01,\n"
        "2020-04-01,4.0\n")
    out = dd.read_fred("f.csv")
    assert out.index.tolist() == [pd.Timestamp("2020-01-01"),
                                  pd.Timestamp("2020-04-01")]
    assert out.tolist() == [1.0, 4.0]


def test_fred_quarterly_averages_months_and_passes_quarters_through(dd):
    (dd.RAW / "m.csv").write_text(
        "observation_date,X\n2020-01-01,1\n2020-02-01,2\n2020-03-01,6\n"
        "2020-04-01,10\n2020-05-01,20\n2020-06-01,30\n")
    (dd.RAW / "q.csv").write_text(
        "observation_date,X\n2020-01-01,7\n2020-04-01,9\n")
    assert dd.fred_quarterly("m.csv").to_dict() == {
        pd.Period("2020Q1", "Q"): 3.0, pd.Period("2020Q2", "Q"): 20.0}
    assert dd.fred_quarterly("q.csv").to_dict() == {
        pd.Period("2020Q1", "Q"): 7.0, pd.Period("2020Q2", "Q"): 9.0}


def test_a_part_quarter_is_averaged_from_the_months_it_has(dd):
    """Records current behaviour; it is not an endorsement.

    ``fred_quarterly`` and the ONS monthly averaging take the mean of
    whatever months a quarter has. A quarter with one month is that month.
    In the 6 Nov 2026 vintage this is live in two places: China's CPI ends
    in April 2025, so its 2025Q2 value is April alone, and euro-area HICP
    starts in December 1996, so the 1996Q4 value the German backcast is
    linked to is December alone. ``scripts/compare_ecb_seasonal.py`` guards
    against this (it keeps only quarters with three months); the build does
    not.
    """
    (dd.RAW / "m.csv").write_text(
        "observation_date,X\n2025-01-01,10\n2025-02-01,20\n2025-03-01,30\n"
        "2025-04-01,100\n")
    out = dd.fred_quarterly("m.csv")
    assert out.loc["2025Q1"] == 20.0
    assert out.loc["2025Q2"] == 100.0     # April only


# ---------------------------------------------------------------- transforms

def test_seasonal_adjust_removes_a_constant_quarterly_pattern(dd):
    idx = _q("2000Q1", "2011Q4")
    trend = _growing(idx, 0.01)
    factors = np.array([1.02, 0.99, 1.01, 0.98])      # mean 1
    raw = trend * factors[idx.quarter - 1]
    out = dd.seasonal_adjust(raw)
    # The estimated factors are the true ones up to the trend's curvature
    # inside the 2x4 moving average (cosh terms of order g^2).
    np.testing.assert_allclose(out.to_numpy(), trend.to_numpy(), rtol=2e-4)
    assert (out / trend).std() < 1e-4


def test_seasonal_adjust_leaves_an_unseasonal_series_alone(dd):
    s = _growing(_q("2000Q1", "2011Q4"), 0.01)
    np.testing.assert_allclose(dd.seasonal_adjust(s).to_numpy(), s.to_numpy(),
                               rtol=1e-12)


def test_seasonal_adjust_tv_removes_a_seasonal_pattern_in_growth(dd):
    """A 100*log level with a fixed seasonal of up to 0.4 points in its
    quarterly change comes back with the trend growth (0.5) to within 0.003
    points, and the first level is kept. The first two changes are the
    exception: the filter has no change for the first observation, so it
    leaves 0.075 points of the pattern there (measured, recorded here)."""
    idx = _q("1990Q1", "2025Q4")
    seasonal = np.array([-0.2, 0.4, -0.3, 0.1])        # sums to zero
    d = 0.5 + seasonal[idx.quarter - 1]
    level = pd.Series(460.0 + np.cumsum(d) - d[0], index=idx)
    out = dd.seasonal_adjust_tv(level)
    assert out.iloc[0] == level.iloc[0]
    change = out.diff().dropna().to_numpy()
    np.testing.assert_allclose(change[2:], 0.5, atol=0.003)
    np.testing.assert_allclose(change[:2], 0.5754, atol=1e-3)


def test_seasonal_adjust_tv_leaves_an_unseasonal_series_alone(dd):
    level = pd.Series(460.0 + 0.5 * np.arange(120), index=_q("1990Q1", "2019Q4"))
    np.testing.assert_allclose(dd.seasonal_adjust_tv(level).to_numpy(),
                               level.to_numpy(), atol=1e-9)


def test_chain_weighted_is_weighted_log_growth_from_base_100(dd):
    idx = _q("2000Q1", "2000Q3")
    a = pd.Series([100.0, 110.0, 121.0], index=idx)     # +10% a quarter
    b = pd.Series([50.0, 50.0, 100.0], index=idx)       # 0, then doubles
    out = dd.chain_weighted([(a, 0.5), (b, 0.5)])
    assert out.index.equals(idx)
    assert out.iloc[0] == 100.0
    assert out.iloc[1] == pytest.approx(100 * 1.1 ** 0.5)
    assert out.iloc[2] == pytest.approx(100 * 1.1 * 2 ** 0.5)


def test_chain_weighted_starts_where_every_component_has_growth(dd):
    a = pd.Series([1.0, 2.0, 4.0, 8.0], index=_q("2000Q1", "2000Q4"))
    b = pd.Series([3.0, 3.0, 3.0], index=_q("2000Q2", "2000Q4"))
    out = dd.chain_weighted([(a, 0.5), (b, 0.5)])
    assert str(out.index[0]) == "2000Q2" and out.iloc[0] == 100.0
    assert out.loc["2000Q3"] == pytest.approx(100 * 2 ** 0.5)


def test_annual_to_quarterly_spans_q3_of_first_year_to_q2_of_last(dd):
    a = pd.Series([100.0, 100.0 * np.exp(0.08), 100.0 * np.exp(0.16)],
                  index=pd.to_datetime(["2000-01-01", "2001-01-01", "2002-01-01"]))
    out = dd.annual_to_quarterly(a)
    assert [str(p) for p in out.index] == [
        "2000Q3", "2000Q4", "2001Q1", "2001Q2", "2001Q3", "2001Q4",
        "2002Q1", "2002Q2"]
    # Annual values sit halfway between Q2 and Q3, so 2000Q3 is an eighth of
    # a year past the first one and each quarter adds a quarter of 0.08.
    assert out.loc["2000Q3"] == pytest.approx(100 * np.exp(0.01))
    np.testing.assert_allclose(np.diff(np.log(out.to_numpy())), 0.02, atol=1e-12)
    assert out.loc["2002Q2"] == pytest.approx(100 * np.exp(0.15))


def test_extend_with_annual_inflation_compounds_to_the_annual_rate(dd):
    q = pd.Series([100.0, 101.0], index=_q("2021Q1", "2021Q2"))
    annual = pd.Series([4.0, 8.0], index=pd.to_datetime(["2021-01-01",
                                                         "2022-01-01"]))
    out = dd.extend_with_annual_inflation(q, annual, "2026Q4")
    # Stops at the last year with an annual rate, not at ``thru``.
    assert str(out.index[-1]) == "2022Q4" and len(out) == 8
    assert out.loc["2021Q3"] == pytest.approx(101.0 * 1.04 ** 0.25)
    assert out.loc["2021Q4"] == pytest.approx(101.0 * 1.04 ** 0.5)
    # Four quarters at the 2022 rate compound to exactly 8%.
    assert out.loc["2022Q4"] / out.loc["2021Q4"] == pytest.approx(1.08)
    assert out.loc[:"2021Q2"].tolist() == [100.0, 101.0]


def test_extend_with_annual_inflation_stops_at_thru(dd):
    q = pd.Series([100.0], index=_q("2021Q4", "2021Q4"))
    annual = pd.Series([2.0, 2.0], index=pd.to_datetime(["2022-01-01",
                                                         "2023-01-01"]))
    out = dd.extend_with_annual_inflation(q, annual, "2022Q2")
    assert [str(p) for p in out.index] == ["2021Q4", "2022Q1", "2022Q2"]


def test_extend_with_annual_inflation_adds_nothing_without_a_rate(dd):
    q = pd.Series([100.0], index=_q("2021Q4", "2021Q4"))
    annual = pd.Series([2.0], index=pd.to_datetime(["2021-01-01"]))
    out = dd.extend_with_annual_inflation(q, annual, "2026Q4")
    assert out.to_dict() == {pd.Period("2021Q4", "Q"): 100.0}


@pytest.mark.parametrize("quarter, era", [
    ("1988Q1", ERA1), ("1990Q1", ERA1), ("1999Q4", ERA1),
    ("2000Q1", ERA2), ("2009Q4", ERA2),
    ("2010Q1", ERA3), ("2026Q4", ERA3), ("2031Q1", ERA3),
])
def test_era_weights_switch_at_the_era_boundaries(dd, quarter, era):
    assert dd.era_weights(pd.Period(quarter, "Q")) == era


def test_era_weights_are_the_documented_table(dd):
    """data/README.md prints these shares; a change here must change both."""
    assert [(a, b) for a, b, _ in dd.TRADE_WEIGHT_ERAS] == [
        ("1990Q1", "1999Q4"), ("2000Q1", "2009Q4"), ("2010Q1", "2026Q4")]
    assert [w for _, _, w in dd.TRADE_WEIGHT_ERAS] == [ERA1, ERA2, ERA3]
    readme = (ROOT / "data" / "README.md").read_text()
    for label, w in (("1992Q1–1999Q4", ERA1), ("2000Q1–2009Q4", ERA2),
                     ("2010Q1–2026Q2", ERA3)):
        row = f"| {label} | {w['ea']:.2f} | {w['us']:.2f} | {w['jp']:.2f} | {w['cn']:.2f} |"
        assert row in readme


def test_chain_weighted_tv_renormalises_over_the_countries_with_data(dd):
    idx = _q("2014Q1", "2016Q4")
    us = _growing(idx, 0.01)
    ea = _growing(idx, 0.02)
    jp = _growing(idx, 0.03).loc[:"2015Q2"]      # drops out after 2015Q2
    cn = _growing(idx, 0.05).loc["2015Q1":]      # growth from 2015Q2
    out = np.log(dd.chain_weighted_tv({"us": us, "ea": ea, "jp": jp, "cn": cn}))
    g = out.diff()
    # 2014Q2: no China. 2015Q2: all four. 2015Q3: no Japan.
    assert g.loc["2014Q2"] == pytest.approx(
        (0.16 * 0.01 + 0.45 * 0.02 + 0.02 * 0.03) / 0.63)
    assert g.loc["2015Q2"] == pytest.approx(
        (0.16 * 0.01 + 0.45 * 0.02 + 0.02 * 0.03 + 0.07 * 0.05) / 0.70)
    assert g.loc["2015Q3"] == pytest.approx(
        (0.16 * 0.01 + 0.45 * 0.02 + 0.07 * 0.05) / 0.68)
    # Rebased so the 2015 average of the index is 100.
    assert np.exp(out.loc["2015Q1":"2015Q4"]).mean() == pytest.approx(100.0)
    assert out.index.equals(idx)


def test_chain_weighted_tv_changes_weights_between_eras(dd):
    idx = _q("1999Q1", "2016Q4")
    legs = {"us": _growing(idx, 0.01), "ea": _growing(idx, 0.02),
            "jp": _growing(idx, 0.03), "cn": _growing(idx, 0.05)}
    g = np.log(dd.chain_weighted_tv(legs)).diff()
    growth = {"us": 0.01, "ea": 0.02, "jp": 0.03, "cn": 0.05}
    assert g.loc["1999Q4"] == pytest.approx(_weighted(ERA1, growth))
    assert g.loc["2000Q1"] == pytest.approx(_weighted(ERA2, growth))
    assert g.loc["2009Q4"] == pytest.approx(_weighted(ERA2, growth))
    assert g.loc["2010Q1"] == pytest.approx(_weighted(ERA3, growth))


def test_splice_back_links_at_the_first_quarter_of_the_main_series(dd):
    main = pd.Series([200.0, 210.0], index=_q("1995Q1", "1995Q2"))
    back = pd.Series([40.0, 44.0, 50.0, 99.0], index=_q("1994Q3", "1995Q2"))
    out = dd.splice_back(main, back)
    # ratio = 200 / 50 = 4; the backcast's later values are ignored.
    assert out.to_dict() == {
        pd.Period("1994Q3", "Q"): 160.0, pd.Period("1994Q4", "Q"): 176.0,
        pd.Period("1995Q1", "Q"): 200.0, pd.Period("1995Q2", "Q"): 210.0}


def test_splice_back_refuses_a_backcast_that_does_not_reach_the_link(dd):
    main = pd.Series([200.0, 210.0], index=_q("1995Q1", "1995Q2"))
    short = pd.Series([40.0, 44.0], index=_q("1994Q2", "1994Q3"))
    with pytest.raises(ValueError, match="does not reach start"):
        dd.splice_back(main, short)
    late = pd.Series([40.0, 44.0], index=_q("1995Q2", "1995Q3"))
    with pytest.raises(ValueError, match="does not reach start"):
        dd.splice_back(main, late)


# -------------------------------------------------- build(): the UK columns

def test_build_columns_and_sample(built):
    dd, df, df_v1 = built
    assert list(df.columns) == ["world_gdp", "world_cpi", "oil_price",
                                "bank_rate", "eri", "cpisa", "cpi_energy",
                                "uk_gdp"]
    # Brent starts in 1992Q1; euro-area and Japanese GDP end in 2026Q2.
    assert str(df.index[0]) == "1992Q1" and str(df.index[-1]) == "2026Q2"
    assert len(df) == 138 and not df.isna().any().any()
    assert list(df_v1.columns) == ["world_gdp", "world_cpi"]
    assert df_v1.index.equals(df.index)


def test_build_uk_gdp_bank_rate_and_exchange_rate(built):
    _, df, _ = built
    n = _quarters_since("1988Q1", "2010Q3")
    assert df.loc["2010Q3", "uk_gdp"] == pytest.approx(
        100 * (np.log(400000.0) + G_UK_GDP * n))
    assert df.loc["2010Q3", "eri"] == pytest.approx(
        100 * (np.log(100.0) + G_ERI * n))
    # Bank Rate is the level in percent, not a log.
    assert df.loc["2010Q3", "bank_rate"] == pytest.approx(5.0 + 0.01 * n)


def test_build_uk_cpi_is_the_quarterly_average_of_the_monthly_index(built):
    """The synthetic CPI has no seasonal pattern, so the constant-factor
    adjustment must leave the quarterly average unchanged."""
    _, df, _ = built
    months = 3 * _quarters_since("1988Q1", "2010Q3") + np.arange(3)
    expected = np.mean(50.0 * np.exp(G_UK_CPI_M * months))
    assert df.loc["2010Q3", "cpisa"] == pytest.approx(100 * np.log(expected))
    np.testing.assert_allclose(df["cpisa"].diff().dropna(), 300 * G_UK_CPI_M,
                               atol=1e-9)


def test_build_cpi_energy_is_the_equal_weight_chain_of_its_two_legs(built):
    _, df, _ = built
    n = _quarters_since("1988Q1", "2010Q3")
    per_quarter = 0.5 * 3 * G_E1_M + 0.5 * 3 * G_E2_M
    assert df.loc["2010Q3", "cpi_energy"] == pytest.approx(
        100 * (np.log(100.0) + per_quarter * n))


def test_build_oil_price_is_brent_in_sterling_deflated_by_uk_cpi(built):
    _, df, _ = built
    brent = 20.0 * np.exp(G_BRENT * _quarters_since("1992Q1", "2010Q3"))
    cpisa = np.exp(df.loc["2010Q3", "cpisa"] / 100)
    assert df.loc["2010Q3", "oil_price"] == pytest.approx(
        100 * np.log(brent / USD_PER_GBP / cpisa * 100.0))


# ---------------------------------------- build(): the world-series proxies

def _g(df, col, quarter):
    return df[col].diff().loc[quarter] / 100.0


def test_world_gdp_before_japan_has_data(built):
    """1993Q1: era-1 weights; the euro area is the German backcast; Japan's
    series starts in 1994Q1, so its first growth rate is 1994Q2."""
    _, df, _ = built
    cn = G_CN_GDP_ANNUAL / 4
    assert _g(df, "world_gdp", "1993Q1") == pytest.approx(
        _weighted(ERA1, {"us": G_US_GDP, "ea": G_DE_GDP, "cn": cn}))
    assert _g(df, "world_gdp", "1994Q1") == pytest.approx(
        _weighted(ERA1, {"us": G_US_GDP, "ea": G_DE_GDP, "cn": cn}))
    assert _g(df, "world_gdp", "1994Q2") == pytest.approx(
        _weighted(ERA1, {"us": G_US_GDP, "ea": G_DE_GDP, "jp": G_JP_GDP,
                         "cn": cn}))


def test_world_gdp_euro_area_leg_switches_from_germany_to_ea19(built):
    """EA19 GDP starts in 1995Q1: growth into 1995Q1 is still Germany's,
    growth into 1995Q2 is the euro area's."""
    _, df, _ = built
    rest = {"us": G_US_GDP, "jp": G_JP_GDP, "cn": G_CN_GDP_ANNUAL / 4}
    assert _g(df, "world_gdp", "1995Q1") == pytest.approx(
        _weighted(ERA1, rest | {"ea": G_DE_GDP}))
    assert _g(df, "world_gdp", "1995Q2") == pytest.approx(
        _weighted(ERA1, rest | {"ea": G_EA_GDP}))


@pytest.mark.parametrize("quarter, era", [("1999Q4", ERA1), ("2000Q1", ERA2),
                                          ("2009Q4", ERA2), ("2010Q1", ERA3)])
def test_world_gdp_uses_the_era_trade_weights(built, quarter, era):
    _, df, _ = built
    assert _g(df, "world_gdp", quarter) == pytest.approx(_weighted(era, {
        "us": G_US_GDP, "ea": G_EA_GDP, "jp": G_JP_GDP,
        "cn": G_CN_GDP_ANNUAL / 4}))


def test_world_gdp_china_is_interpolated_then_drops_out_after_2023q2(built):
    """The annual Chinese series ends in 2023. Mid-year placement takes the
    quarterly path to 2023Q2; from 2023Q3 the weights are renormalised over
    the United States, the euro area and Japan."""
    _, df, _ = built
    three = {"us": G_US_GDP, "ea": G_EA_GDP, "jp": G_JP_GDP}
    assert _g(df, "world_gdp", "2023Q2") == pytest.approx(
        _weighted(ERA3, three | {"cn": G_CN_GDP_ANNUAL / 4}))
    assert _g(df, "world_gdp", "2023Q3") == pytest.approx(_weighted(ERA3, three))
    assert _g(df, "world_gdp", "2026Q2") == pytest.approx(_weighted(ERA3, three))


def test_world_series_are_rebased_to_2015(built):
    _, df, _ = built
    for col in ("world_gdp", "world_cpi"):
        assert np.exp(df.loc["2015Q1":"2015Q4", col] / 100).mean() == \
            pytest.approx(100.0)


def test_world_cpi_before_china_has_data(built):
    """1992Q2: the euro area is the German backcast and China's CPI starts
    in January 1993 (first growth rate 1993Q2)."""
    _, df, _ = built
    three = {"us": G_US_CPI_OECD, "ea": 3 * G_DE_CPI_M, "jp": G_JP_CPI}
    assert _g(df, "world_cpi", "1992Q2") == pytest.approx(_weighted(ERA1, three))
    assert _g(df, "world_cpi", "1993Q1") == pytest.approx(_weighted(ERA1, three))
    assert _g(df, "world_cpi", "1993Q2") == pytest.approx(
        _weighted(ERA1, three | {"cn": 3 * G_CN_CPI_M}))


def test_world_cpi_euro_area_leg_switches_from_germany_to_hicp(built):
    """HICP starts in 1997Q1 in the synthetic files: growth into 1997Q1 is
    German CPI's, growth into 1997Q2 is HICP's."""
    _, df, _ = built
    rest = {"us": G_US_CPI_OECD, "jp": G_JP_CPI, "cn": 3 * G_CN_CPI_M}
    assert _g(df, "world_cpi", "1997Q1") == pytest.approx(
        _weighted(ERA1, rest | {"ea": 3 * G_DE_CPI_M}))
    assert _g(df, "world_cpi", "1997Q2") == pytest.approx(
        _weighted(ERA1, rest | {"ea": 3 * G_EA_CPI_M}))


@pytest.mark.parametrize("quarter, era", [("1999Q4", ERA1), ("2000Q1", ERA2),
                                          ("2009Q4", ERA2), ("2010Q1", ERA3)])
def test_world_cpi_uses_the_era_trade_weights(built, quarter, era):
    _, df, _ = built
    assert _g(df, "world_cpi", quarter) == pytest.approx(_weighted(era, {
        "us": G_US_CPI_OECD, "ea": 3 * G_EA_CPI_M, "jp": G_JP_CPI,
        "cn": 3 * G_CN_CPI_M}))


def test_world_cpi_japan_follows_annual_inflation_after_2021q2(built):
    """The quarterly Japanese series ends in 2021Q2; later quarters grow at
    a quarter of the log of the annual rate."""
    _, df, _ = built
    rest = {"us": G_US_CPI_OECD, "ea": 3 * G_EA_CPI_M, "cn": 3 * G_CN_CPI_M}
    assert _g(df, "world_cpi", "2021Q2") == pytest.approx(
        _weighted(ERA3, rest | {"jp": G_JP_CPI}))
    assert _g(df, "world_cpi", "2021Q3") == pytest.approx(
        _weighted(ERA3, rest | {"jp": np.log(1 + JP_ANNUAL_PCT / 100) / 4}))


def test_world_cpi_us_leg_is_spliced_to_bls_growth_after_2025q1(built):
    """The OECD US series ends in 2025Q1. From 2025Q2 the US leg grows with
    CPIAUCSL (three times its monthly rate), linked at 2025Q1. China's last
    month is April 2025, so its 2025Q2 'quarter' is April over the 2025Q1
    average (see test_a_part_quarter_is_averaged_from_the_months_it_has)."""
    _, df, _ = built
    jp = np.log(1 + JP_ANNUAL_PCT / 100) / 4
    r = G_CN_CPI_M
    # April over the January-March mean: e^{2r} * 3 / (1 + 2 cosh r).
    cn_april_only = 2 * r + np.log(3 / (1 + 2 * np.cosh(r)))
    assert _g(df, "world_cpi", "2025Q1") == pytest.approx(_weighted(ERA3, {
        "us": G_US_CPI_OECD, "ea": 3 * G_EA_CPI_M, "jp": jp, "cn": 3 * r}))
    assert _g(df, "world_cpi", "2025Q2") == pytest.approx(_weighted(ERA3, {
        "us": 3 * G_US_CPI_BLS_M, "ea": 3 * G_EA_CPI_M, "jp": jp,
        "cn": cn_april_only}))
    assert cn_april_only == pytest.approx(2 * r, abs=1e-5)   # not 3r


def test_world_cpi_china_then_japan_drop_out_at_the_end(built):
    """As in the 6 Nov 2026 vintage: China leaves world CPI after 2025Q2,
    and Japan after 2025Q4 (annual inflation ends in 2025). The weights are
    renormalised each time."""
    _, df, _ = built
    us, ea = 3 * G_US_CPI_BLS_M, 3 * G_EA_CPI_M
    jp = np.log(1 + JP_ANNUAL_PCT / 100) / 4
    assert _g(df, "world_cpi", "2025Q3") == pytest.approx(
        _weighted(ERA3, {"us": us, "ea": ea, "jp": jp}))
    assert _g(df, "world_cpi", "2025Q4") == pytest.approx(
        _weighted(ERA3, {"us": us, "ea": ea, "jp": jp}))
    assert _g(df, "world_cpi", "2026Q1") == pytest.approx(
        _weighted(ERA3, {"us": us, "ea": ea}))
    assert _g(df, "world_cpi", "2026Q2") == pytest.approx(
        _weighted(ERA3, {"us": us, "ea": ea}))


def test_v1_proxies_are_two_sevenths_us_and_five_sevenths_euro_area(built):
    _, _, df_v1 = built
    assert df_v1["world_gdp"].diff().loc["2005Q1"] / 100 == pytest.approx(
        (2 * G_US_GDP + 5 * G_EA_GDP) / 7)
    assert df_v1["world_cpi"].diff().loc["2005Q1"] / 100 == pytest.approx(
        (2 * G_US_CPI_OECD + 5 * 3 * G_EA_CPI_M) / 7)
    # Before EA19 GDP starts, the euro-area leg is the German backcast.
    assert df_v1["world_gdp"].diff().loc["1993Q1"] / 100 == pytest.approx(
        (2 * G_US_GDP + 5 * G_DE_GDP) / 7)


# ------------------------------------------- build(): the edge of the sample

def test_the_shortest_series_sets_the_last_quarter(dd):
    """UK GDP for a quarter arrives after most other series. If it ends a
    quarter early, so does the dataset."""
    write_raw(dd.RAW, ons_abmi=_growing(_q("1988Q1", "2026Q1"), G_UK_GDP, 4e5))
    df, _ = dd.build()
    assert str(df.index[-1]) == "2026Q1"


def test_a_series_that_ends_a_month_early_drops_the_whole_quarter(dd):
    """No: it does not. This records current behaviour.

    UK CPI to August 2026 gives a 2026Q3 value from two months. The dataset
    does not show it only because another series (here, as in the 6 Nov 2026
    vintage, euro-area and Japanese GDP) ends in 2026Q2 and the script caps
    the sample at 2026Q2.
    """
    write_raw(dd.RAW, ons_d7bt=_growing(_m("1988-01", "2026-08"), G_UK_CPI_M,
                                        50.0))
    df, _ = dd.build()
    assert str(df.index[-1]) == "2026Q2"


def test_the_sample_is_capped_at_2026q2_even_when_every_series_runs_on(dd):
    """The cap is a literal in the script, so a refresh for 2026Q3 needs a
    code change as well as new downloads."""
    later = {
        "ons_abmi": _growing(_q("1988Q1", "2026Q4"), G_UK_GDP, 4e5),
        "ons_d7bt": _growing(_m("1988-01", "2026-12"), G_UK_CPI_M, 50.0),
        "ons_d7ch": _growing(_m("1988-01", "2026-12"), G_E1_M, 40.0),
        "ons_d7ec": _growing(_m("1988-01", "2026-12"), G_E2_M, 30.0),
        "boe_bank_rate": pd.Series(5.0, index=_q("1988Q1", "2026Q4")),
        "boe_eri": _growing(_q("1988Q1", "2026Q4"), G_ERI, 100.0),
        "boe_usd": pd.Series(USD_PER_GBP, index=_q("1988Q1", "2026Q4")),
        "fred_POILBREUSDQ": _growing(_q("1992Q1", "2026Q4"), G_BRENT, 20.0),
        "fred_GDPC1": _growing(_q("1988Q1", "2026Q4"), G_US_GDP, 9000.0),
        "fred_CLVMNACSCAB1GQEA19": _growing(_q("1995Q1", "2026Q4"), G_EA_GDP,
                                            2e6),
        "fred_CLVMNACSCAB1GQDE": _growing(_q("1988Q1", "2026Q4"), G_DE_GDP, 5e5),
        "fred_JPNRGDPEXP": _growing(_q("1994Q1", "2026Q4"), G_JP_GDP, 4.5e5),
        "fred_CPIAUCSL": _growing(_m("1988-01", "2026-12"), G_US_CPI_BLS_M,
                                  115.0),
        "fred_CP0000EZ19M086NEST": _growing(_m("1997-01", "2026-12"),
                                            G_EA_CPI_M, 70.0),
    }
    write_raw(dd.RAW, **later)
    df, _ = dd.build()
    assert str(df.index[-1]) == "2026Q2"


def test_a_late_start_trims_the_sample_and_fails_the_window_check(dd):
    """Brent sets the first quarter. If it started in 1993 the 1992Q1-2023Q2
    window would be incomplete and the build must stop."""
    write_raw(dd.RAW, fred_POILBREUSDQ=_growing(_q("1993Q1", "2026Q3"),
                                                G_BRENT, 20.0))
    with pytest.raises(AssertionError, match="required window incomplete"):
        dd.build()


def test_the_window_check_stops_at_2023q2_not_at_the_estimation_end(dd):
    """Records current behaviour. data/README.md says the script asserts
    that 1992Q1-2025Q1 (the estimation window) is complete; the assertion
    in the script covers 1992Q1-2023Q2. A dataset that ends in 2024Q4 still
    builds."""
    write_raw(dd.RAW, ons_abmi=_growing(_q("1988Q1", "2024Q4"), G_UK_GDP, 4e5))
    df, _ = dd.build()
    assert str(df.index[-1]) == "2024Q4"


# ------------------------------------------------------- finalise and main

def test_finalise_adjusts_world_cpi_only_and_keeps_the_unadjusted_series(built):
    dd, df, _ = built
    out, nsa = dd.finalise(df)
    assert list(nsa.columns) == ["world_cpi"]
    pd.testing.assert_series_equal(nsa["world_cpi"], df["world_cpi"])
    pd.testing.assert_series_equal(out["world_cpi"],
                                   dd.seasonal_adjust_tv(df["world_cpi"]),
                                   check_names=False)
    pd.testing.assert_frame_equal(out.drop(columns="world_cpi"),
                                  df.drop(columns="world_cpi"))
    assert out["world_cpi"].iloc[0] == df["world_cpi"].iloc[0]


def test_main_writes_the_four_files_under_root_and_touches_no_network(
        dd, monkeypatch, capsys):
    write_raw(dd.RAW)

    def no_get(*args, **kwargs):
        raise AssertionError("main() tried to download a cached file")

    monkeypatch.setattr(dd.requests, "get", no_get)
    assert dd.main() is None
    capsys.readouterr()

    out = pd.read_csv(dd.OUT)
    assert list(out.columns) == ["quarter", "world_gdp", "world_cpi",
                                 "oil_price", "bank_rate", "eri", "cpisa",
                                 "cpi_energy", "uk_gdp"]
    assert out["quarter"].iloc[0] == "1992Q1"
    assert out["quarter"].iloc[-1] == "2026Q2" and len(out) == 138
    # Six decimals, as committed.
    assert all(len(v.split(".")[1]) == 6
               for v in dd.OUT.read_text().splitlines()[1].split(",")[1:])

    packaged = dd.ROOT / "src" / "boe_var" / "_data" / "boe_var_data.csv"
    assert packaged.read_bytes() == dd.OUT.read_bytes()

    nsa = pd.read_csv(dd.OUT.parent / "boe_var_data_world_cpi_nsa.csv")
    assert list(nsa.columns) == ["quarter", "world_cpi"]
    assert nsa["quarter"].tolist() == out["quarter"].tolist()
    nsa.index = pd.PeriodIndex(nsa["quarter"], freq="Q")
    adjusted = dd.seasonal_adjust_tv(nsa["world_cpi"])
    np.testing.assert_allclose(adjusted.to_numpy(), out["world_cpi"].to_numpy(),
                               atol=2e-6)

    v1 = pd.read_csv(dd.OUT.parent / "boe_var_data_v1_us_ea_only.csv")
    assert list(v1.columns) == ["quarter", "world_gdp", "world_cpi"]
    assert v1["quarter"].tolist() == out["quarter"].tolist()


def test_the_test_module_cannot_write_to_the_repository(dd):
    assert ROOT not in dd.OUT.parents and ROOT not in dd.RAW.parents
    assert dd.ROOT != ROOT


# -------------------------------------------------------------- download_all

class _Response:
    def __init__(self, text: str, status: int = 200):
        self.text, self.status_code, self.ok = text, status, status < 400


def _stub_get(dd, monkeypatch, reply):
    calls = []

    def get(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        return reply(url, len(calls))

    monkeypatch.setattr(dd.requests, "get", get)
    return calls


def test_download_all_fetches_each_source_once_and_writes_it(dd, monkeypatch,
                                                           capsys):
    body = "observation_date,X\n" + "2020-01-01,1.0\n" * 20
    calls = _stub_get(dd, monkeypatch, lambda url, n: _Response(body))
    dd.download_all()
    capsys.readouterr()
    assert [c["url"] for c in calls] == list(dd.DOWNLOADS.values())
    assert len(calls) == 18
    assert all(c["timeout"] == 60 and "User-Agent" in c["headers"]
               for c in calls)
    assert sorted(p.name for p in dd.RAW.iterdir()) == sorted(dd.DOWNLOADS)
    assert all(p.read_text() == body for p in dd.RAW.iterdir())
    # Eight seconds after each of the four ONS files, one after the others.
    assert dd.slept == [8, 8, 8, 8] + [1] * 14


def test_download_all_reuses_cached_files(dd, monkeypatch, capsys):
    write_raw(dd.RAW)
    before = {p.name: p.read_bytes() for p in dd.RAW.iterdir()}
    calls = _stub_get(dd, monkeypatch, lambda url, n: _Response("x" * 500))
    dd.download_all()
    capsys.readouterr()
    assert calls == [] and dd.slept == []
    assert {p.name: p.read_bytes() for p in dd.RAW.iterdir()} == before


def test_download_all_refetches_a_stub_sized_cache_file(dd, monkeypatch, capsys):
    """A cached file of 200 bytes or fewer is treated as a failed download."""
    write_raw(dd.RAW)
    (dd.RAW / "fred_GDPC1.csv").write_text("truncated")
    body = "observation_date,GDPC1\n" + "2020-01-01,1.0\n" * 20
    calls = _stub_get(dd, monkeypatch, lambda url, n: _Response(body))
    dd.download_all()
    capsys.readouterr()
    assert [c["url"] for c in calls] == [dd.DOWNLOADS["fred_GDPC1.csv"]]
    assert (dd.RAW / "fred_GDPC1.csv").read_text() == body


def test_download_all_retries_an_html_reply_and_never_caches_it(
        dd, monkeypatch, capsys):
    """ONS answers a rate-limited request with an HTML page and status 200.
    It must not be written to data/raw as if it were the series."""
    write_raw(dd.RAW)
    (dd.RAW / "ons_abmi.csv").unlink()
    good = _ons_text("ABMI", quarterly=raw_series()["ons_abmi"])
    replies = [_Response("<!DOCTYPE html><html>rate limited</html>"),
               _Response("\n  <html>busy</html>"),
               _Response("error", status=503),
               _Response(good)]
    calls = _stub_get(dd, monkeypatch, lambda url, n: replies[n - 1])
    dd.download_all()
    capsys.readouterr()
    assert len(calls) == 4
    assert (dd.RAW / "ons_abmi.csv").read_text() == good
    assert dd.slept == [30, 60, 90, 8]      # back-off, then the ONS pause


def test_download_all_gives_up_after_four_failures(dd, monkeypatch, capsys):
    write_raw(dd.RAW)
    (dd.RAW / "boe_rates_fx.csv").unlink()
    calls = _stub_get(dd, monkeypatch,
                      lambda url, n: _Response("<html>no</html>", status=403))
    with pytest.raises(RuntimeError, match="failed to download boe_rates_fx.csv"):
        dd.download_all()
    capsys.readouterr()
    assert len(calls) == 4 and dd.slept == [30, 60, 90, 120]
    assert not (dd.RAW / "boe_rates_fx.csv").exists()


def test_download_urls_name_the_documented_series(dd):
    """Each URL asks its publisher for the series data/README.md lists."""
    d = dd.DOWNLOADS
    assert len(d) == 18
    for name, cdid, dataset in (("ons_abmi.csv", "abmi", "ukea"),
                                ("ons_d7bt.csv", "d7bt", "mm23"),
                                ("ons_d7ch.csv", "d7ch", "mm23"),
                                ("ons_d7ec.csv", "d7ec", "mm23")):
        assert d[name].startswith("https://www.ons.gov.uk/generator?format=csv")
        assert d[name].endswith(f"/timeseries/{cdid}/{dataset}")
    assert d["boe_rates_fx.csv"].startswith("https://www.bankofengland.co.uk/")
    assert "SeriesCodes=IUQABEDR,XUQABK67,XUQAUSS" in d["boe_rates_fx.csv"]
    readme = (ROOT / "data" / "README.md").read_text()
    for name, url in d.items():
        if name.startswith("fred_"):
            sid = name.removeprefix("fred_").removesuffix(".csv")
            assert url == f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            assert f"`{sid}`" in readme, f"{sid} is not documented in data/README.md"

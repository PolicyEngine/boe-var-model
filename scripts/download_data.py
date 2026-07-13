#!/usr/bin/env python
"""Download raw sources and build data/boe_var_data.csv for the BoE SVAR replication.

Raw files are cached in data/raw/ (delete a file to force re-download).
Final CSV columns (SPEC.md): quarter, world_gdp, world_cpi, oil_price, bank_rate,
eri, cpisa, cpi_energy, uk_gdp — already transformed (100*log except bank_rate).

Sources:
- ONS time-series CSV generator (ABMI, D7BT, D7CH, D7EC)
- Bank of England IADB CSV (IUQABEDR, XUQABK67, XUQAUSS)
- FRED CSV (POILBREUSDQ, GDPC1, CLVMNACSCAB1GQEA19, CLVMNACSCAB1GQDE,
  CPALTT01USQ661S, CP0000EZ19M086NEST, DEUCPIALLMINMEI)

See data/README.md for full documentation of series and proxy decisions.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "boe_var_data.csv"

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) boe-var-replication"}

ONS_URL = "https://www.ons.gov.uk/generator?format=csv&uri=/economy/{area}/timeseries/{cdid}/{dataset}"
BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes"
    "&Datefrom=01/Jan/1988&Dateto=01/Oct/2024&SeriesCodes={codes}"
    "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

DOWNLOADS = {
    # name -> url
    "ons_abmi.csv": ONS_URL.format(area="grossdomesticproductgdp", cdid="abmi", dataset="ukea"),
    "ons_d7bt.csv": ONS_URL.format(area="inflationandpriceindices", cdid="d7bt", dataset="mm23"),
    "ons_d7ch.csv": ONS_URL.format(area="inflationandpriceindices", cdid="d7ch", dataset="mm23"),
    "ons_d7ec.csv": ONS_URL.format(area="inflationandpriceindices", cdid="d7ec", dataset="mm23"),
    "boe_rates_fx.csv": BOE_URL.format(codes="IUQABEDR,XUQABK67,XUQAUSS"),
    "fred_POILBREUSDQ.csv": FRED_URL.format(sid="POILBREUSDQ"),
    "fred_GDPC1.csv": FRED_URL.format(sid="GDPC1"),
    "fred_CLVMNACSCAB1GQEA19.csv": FRED_URL.format(sid="CLVMNACSCAB1GQEA19"),
    "fred_CLVMNACSCAB1GQDE.csv": FRED_URL.format(sid="CLVMNACSCAB1GQDE"),
    "fred_CPALTT01USQ661S.csv": FRED_URL.format(sid="CPALTT01USQ661S"),
    "fred_CP0000EZ19M086NEST.csv": FRED_URL.format(sid="CP0000EZ19M086NEST"),
    "fred_DEUCPIALLMINMEI.csv": FRED_URL.format(sid="DEUCPIALLMINMEI"),
    # world aggregate extras: Japan + China
    "fred_JPNRGDPEXP.csv": FRED_URL.format(sid="JPNRGDPEXP"),
    "fred_RGDPNACNA666NRUG.csv": FRED_URL.format(sid="RGDPNACNA666NRUG"),
    "fred_CPALTT01JPQ661S.csv": FRED_URL.format(sid="CPALTT01JPQ661S"),
    "fred_FPCPITOTLZGJPN.csv": FRED_URL.format(sid="FPCPITOTLZGJPN"),
    "fred_CHNCPIALLMINMEI.csv": FRED_URL.format(sid="CHNCPIALLMINMEI"),
}

# UK trade shares (goods+services, approximate era averages, ONS Pink Book /
# IMF DOTS reasoning — see data/README.md). Raw shares of total UK trade; the
# aggregator renormalises over whichever countries have data in each quarter.
TRADE_WEIGHT_ERAS = [
    # (start, end, {country: raw UK trade share})
    ("1990Q1", "1999Q4", {"ea": 0.50, "us": 0.17, "jp": 0.04, "cn": 0.01}),
    ("2000Q1", "2009Q4", {"ea": 0.48, "us": 0.16, "jp": 0.03, "cn": 0.05}),
    ("2010Q1", "2024Q4", {"ea": 0.45, "us": 0.16, "jp": 0.02, "cn": 0.07}),
]


def download_all() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        path = RAW / name
        if path.exists() and path.stat().st_size > 200:
            print(f"[cache] {name}")
            continue
        for attempt in range(4):
            r = requests.get(url, headers=UA, timeout=60)
            text = r.text
            if r.ok and not text.lstrip().lower().startswith(("<!doctype", "<html")):
                path.write_text(text)
                print(f"[dl]    {name} ({len(text)} bytes)")
                break
            wait = 30 * (attempt + 1)
            print(f"[retry] {name}: HTTP {r.status_code} / html response; sleeping {wait}s")
            time.sleep(wait)
        else:
            raise RuntimeError(f"failed to download {name} from {url}")
        time.sleep(8 if "ons.gov.uk" in url else 1)  # be gentle with ONS rate limits


# ---------------------------------------------------------------- parsers

def read_ons(name: str, freq: str) -> pd.Series:
    """Parse an ONS generator CSV; return series indexed by PeriodIndex.

    freq='Q' keeps quarterly rows ('1990 Q1'); freq='M' keeps monthly rows
    ('1990 JAN') and returns a monthly PeriodIndex.
    """
    raw = (RAW / name).read_text().splitlines()
    rows = []
    for line in raw:
        parts = [p.strip('"') for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        label, val = parts[0], parts[1]
        try:
            v = float(val)
        except ValueError:
            continue
        if freq == "Q" and " Q" in label:
            rows.append((pd.Period(label.replace(" ", ""), freq="Q"), v))
        elif freq == "M" and len(label.split()) == 2 and label.split()[1].isalpha():
            yr, mon = label.split()
            try:
                rows.append((pd.Period(f"{yr}-{mon[:3]}", freq="M"), v))
            except Exception:
                continue
    s = pd.Series(dict(rows)).sort_index()
    if s.empty:
        raise ValueError(f"no {freq} rows parsed from {name}")
    return s


def read_boe(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / name)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d %b %Y")
    df = df.set_index(df["DATE"].dt.to_period("Q")).drop(columns="DATE")
    return df.astype(float)


def read_fred(name: str) -> pd.Series:
    df = pd.read_csv(RAW / name)
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"]


def fred_quarterly(name: str) -> pd.Series:
    """FRED series -> quarterly PeriodIndex (averaging if monthly/daily)."""
    s = read_fred(name)
    q = s.groupby(s.index.to_period("Q")).mean()
    return q


# ---------------------------------------------------------- transformations

def seasonal_adjust(q: pd.Series) -> pd.Series:
    """Simple multiplicative seasonal adjustment for a quarterly index.

    Ratio-to-centered-moving-average with constant quarter factors
    (normalised to average 1). X-13-free; documented in data/README.md.
    """
    ma = q.rolling(4, center=True).mean()
    ma = (ma + ma.shift(-1)) / 2  # 2x4 centered MA
    ratio = (q / ma).dropna()
    factors = ratio.groupby(ratio.index.quarter).mean()
    factors /= factors.mean()
    return q / q.index.quarter.map(factors)


def chain_weighted(components: list[tuple[pd.Series, float]]) -> pd.Series:
    """Chain-linked weighted aggregate: index built from weighted log growth."""
    growth = None
    for s, w in components:
        g = np.log(s).diff() * w
        growth = g if growth is None else growth.add(g)
    growth = growth.dropna()
    idx = 100 * np.exp(growth.cumsum())
    # prepend base period
    base = pd.Series([100.0], index=[growth.index[0] - 1])
    return pd.concat([base, idx]).sort_index()


def annual_to_quarterly(a: pd.Series) -> pd.Series:
    """Annual index -> quarterly by log-linear interpolation.

    Each annual value is treated as the mid-year (Q2/Q3 boundary) log level;
    quarterly log levels are interpolated linearly between mid-years. The
    result spans Q3 of the first year to Q2 of the last year (no
    extrapolation beyond published annual data).
    """
    a = a.sort_index()
    years = a.index.year if hasattr(a.index, "year") else [p.year for p in a.index]
    # position annual value halfway between Q2 and Q3 of its year
    x = np.array([pd.Period(f"{y}Q1", freq="Q").ordinal + 1.5 for y in years])
    logs = np.log(a.to_numpy(dtype=float))
    q_start = int(np.ceil(x[0]))
    q_end = int(np.floor(x[-1]))
    qs = np.arange(q_start, q_end + 1)
    interp = np.interp(qs, x, logs)
    idx = pd.PeriodIndex([pd.Period(ordinal=int(o), freq="Q") for o in qs], freq="Q")
    return pd.Series(np.exp(interp), index=idx)


def extend_with_annual_inflation(q: pd.Series, annual_pct: pd.Series, thru: str) -> pd.Series:
    """Extend a quarterly CPI level forward using annual inflation rates.

    Quarters after the last observation grow at (1 + pi_y/100)^(1/4), where
    pi_y is the published annual average inflation rate for that calendar
    year. An approximation (flat within-year profile); documented in README.
    """
    pi = {p.year: v for p, v in annual_pct.items()}
    q = q.sort_index()
    p = q.index[-1]
    vals, idx = [], []
    while p < pd.Period(thru, freq="Q"):
        p = p + 1
        if p.year not in pi:
            break
        g = (1.0 + pi[p.year] / 100.0) ** 0.25
        vals.append((vals[-1] if vals else q.iloc[-1]) * g)
        idx.append(p)
    return pd.concat([q, pd.Series(vals, index=pd.PeriodIndex(idx, freq="Q"))])


def era_weights(period: pd.Period) -> dict[str, float]:
    for start, end, w in TRADE_WEIGHT_ERAS:
        if period <= pd.Period(end, freq="Q"):
            return w  # first era also covers pre-1990 quarters (trimmed later)
    return TRADE_WEIGHT_ERAS[-1][2]


def chain_weighted_tv(components: dict[str, pd.Series]) -> pd.Series:
    """UK-trade-weighted chained aggregate with time-varying era weights.

    world growth_t = sum_i w_it * dlog(x_it), with w_it the era trade shares
    renormalised over the components that have data in both t-1 and t.
    Cumulated into an index, base 100 at the first quarter where any
    component growth exists.
    """
    growth_df = pd.DataFrame({k: np.log(s).diff() for k, s in components.items()})
    growth_df = growth_df.dropna(how="all").sort_index()
    agg = {}
    for p, row in growth_df.iterrows():
        w = era_weights(p)
        avail = [k for k in w if pd.notna(row.get(k))]
        tot = sum(w[k] for k in avail)
        agg[p] = sum(w[k] / tot * row[k] for k in avail)
    g = pd.Series(agg).sort_index()
    idx = np.exp(g.cumsum())
    first = pd.Series([1.0], index=[g.index[0] - 1])
    idx = pd.concat([first, idx]).sort_index()
    # rebase: average of 2015 = 100
    return 100 * idx / idx.loc["2015Q1":"2015Q4"].mean()


def splice_back(main: pd.Series, back: pd.Series) -> pd.Series:
    """Extend `main` backwards using growth rates of `back`."""
    start = main.index[0]
    back = back[back.index <= start]
    if back.empty or back.index[-1] != start:
        raise ValueError("backcast series does not reach start of main series")
    ratio = main.iloc[0] / back.iloc[-1]
    return pd.concat([back.iloc[:-1] * ratio, main]).sort_index()


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    # --- UK GDP
    uk_gdp = read_ons("ons_abmi.csv", "Q")

    # --- UK CPI: monthly NSA -> quarterly average -> simple SA
    cpi_m = read_ons("ons_d7bt.csv", "M")
    cpi_q = cpi_m.groupby(cpi_m.index.asfreq("Q")).mean()
    cpisa = seasonal_adjust(cpi_q)

    # --- UK CPI energy: 04.5 electricity/gas/fuels + 07.2.2 fuels & lubricants,
    #     equal-weight chained aggregate, quarterly averages of monthly indices, NSA
    e1_m = read_ons("ons_d7ch.csv", "M")
    e2_m = read_ons("ons_d7ec.csv", "M")
    e1 = e1_m.groupby(e1_m.index.asfreq("Q")).mean()
    e2 = e2_m.groupby(e2_m.index.asfreq("Q")).mean()
    cpi_energy = chain_weighted([(e1, 0.5), (e2, 0.5)])

    # --- BoE: Bank Rate, ERI, GBP/USD
    boe = read_boe("boe_rates_fx.csv")
    bank_rate = boe["IUQABEDR"]
    eri = boe["XUQABK67"]
    usd_per_gbp = boe["XUQAUSS"]

    # --- Real oil price in sterling: Brent USD / (USD per GBP) deflated by UK CPI
    brent_usd = fred_quarterly("fred_POILBREUSDQ.csv")
    oil_gbp = brent_usd / usd_per_gbp
    oil_real = (oil_gbp / cpisa) * 100.0

    # --- World GDP: UK-trade-weighted US + EA + Japan + China (see README)
    us_gdp = fred_quarterly("fred_GDPC1.csv")
    ea_gdp = fred_quarterly("fred_CLVMNACSCAB1GQEA19.csv")
    de_gdp = fred_quarterly("fred_CLVMNACSCAB1GQDE.csv")
    ea_gdp = splice_back(ea_gdp, de_gdp)
    jp_gdp = fred_quarterly("fred_JPNRGDPEXP.csv")  # 1994Q1-
    cn_gdp = annual_to_quarterly(read_fred("fred_RGDPNACNA666NRUG.csv"))  # PWT annual
    world_gdp = chain_weighted_tv(
        {"us": us_gdp, "ea": ea_gdp, "jp": jp_gdp, "cn": cn_gdp}
    )

    # --- World CPI: same trade weights, US + EA + Japan + China
    us_cpi = fred_quarterly("fred_CPALTT01USQ661S.csv")
    ea_cpi = fred_quarterly("fred_CP0000EZ19M086NEST.csv")
    de_cpi = fred_quarterly("fred_DEUCPIALLMINMEI.csv")
    ea_cpi = splice_back(ea_cpi, de_cpi)
    jp_cpi = fred_quarterly("fred_CPALTT01JPQ661S.csv")  # ends 2021Q2 (OECD exit)
    jp_cpi = extend_with_annual_inflation(
        jp_cpi, read_fred("fred_FPCPITOTLZGJPN.csv"), "2024Q4"
    )
    cn_cpi = fred_quarterly("fred_CHNCPIALLMINMEI.csv")  # 1993M1-
    world_cpi = chain_weighted_tv(
        {"us": us_cpi, "ea": ea_cpi, "jp": jp_cpi, "cn": cn_cpi}
    )

    # --- v1 proxies (US 2/7 + EA 5/7) kept for comparison
    world_gdp_v1 = chain_weighted([(us_gdp, 0.2 / 0.7), (ea_gdp, 0.5 / 0.7)])
    world_cpi_v1 = chain_weighted([(us_cpi, 0.2 / 0.7), (ea_cpi, 0.5 / 0.7)])

    df = pd.DataFrame(
        {
            "world_gdp": 100 * np.log(world_gdp),
            "world_cpi": 100 * np.log(world_cpi),
            "oil_price": 100 * np.log(oil_real),
            "bank_rate": bank_rate,
            "eri": 100 * np.log(eri),
            "cpisa": 100 * np.log(cpisa),
            "cpi_energy": 100 * np.log(cpi_energy),
            "uk_gdp": 100 * np.log(uk_gdp),
        }
    )
    df = df.loc["1990Q1":"2024Q2"].dropna()
    # verify required window
    req = df.loc["1992Q1":"2023Q2"]
    n_req = (pd.Period("2023Q2") - pd.Period("1992Q1")).n + 1
    assert len(req) == n_req and not req.isna().any().any(), (
        f"required window incomplete: {len(req)}/{n_req} rows, "
        f"NaNs={int(req.isna().sum().sum())}"
    )
    df_v1 = pd.DataFrame(
        {
            "world_gdp": 100 * np.log(world_gdp_v1),
            "world_cpi": 100 * np.log(world_cpi_v1),
        }
    ).loc[df.index]
    return df, df_v1


def main() -> None:
    download_all()
    df, df_v1 = build()
    out = df.copy()
    out.insert(0, "quarter", out.index.astype(str))
    out.to_csv(OUT, index=False, float_format="%.6f")
    out_v1 = df_v1.copy()
    out_v1.insert(0, "quarter", out_v1.index.astype(str))
    out_v1.to_csv(OUT.parent / "boe_var_data_v1_us_ea_only.csv", index=False, float_format="%.6f")
    print(f"\nWrote {OUT}: {len(df)} rows, {df.index[0]}–{df.index[-1]}")
    for col in ("world_gdp", "world_cpi"):
        r_lvl = df[col].corr(df_v1[col])
        r_gr = df[col].diff().corr(df_v1[col].diff())
        print(f"  {col}: corr(new,v1) levels={r_lvl:.4f}, growth={r_gr:.4f}")
    print(df.head(3))
    print(df.tail(3))


if __name__ == "__main__":
    sys.exit(main())

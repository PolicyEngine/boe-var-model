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
}


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


def splice_back(main: pd.Series, back: pd.Series) -> pd.Series:
    """Extend `main` backwards using growth rates of `back`."""
    start = main.index[0]
    back = back[back.index <= start]
    if back.empty or back.index[-1] != start:
        raise ValueError("backcast series does not reach start of main series")
    ratio = main.iloc[0] / back.iloc[-1]
    return pd.concat([back.iloc[:-1] * ratio, main]).sort_index()


def build() -> pd.DataFrame:
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

    # --- World GDP proxy: US (GDPC1) + Euro Area (EA19, backcast pre-1995 with DE)
    us_gdp = fred_quarterly("fred_GDPC1.csv")
    ea_gdp = fred_quarterly("fred_CLVMNACSCAB1GQEA19.csv")
    de_gdp = fred_quarterly("fred_CLVMNACSCAB1GQDE.csv")
    ea_gdp = splice_back(ea_gdp, de_gdp)
    # UK trade shares US 0.2, EA 0.5 renormalised (no reliable "other" series)
    world_gdp = chain_weighted([(us_gdp, 0.2 / 0.7), (ea_gdp, 0.5 / 0.7)])

    # --- World CPI proxy: US CPI (SA) + EA HICP (backcast pre-1997 with DE CPI)
    us_cpi = fred_quarterly("fred_CPALTT01USQ661S.csv")
    ea_cpi = fred_quarterly("fred_CP0000EZ19M086NEST.csv")
    de_cpi = fred_quarterly("fred_DEUCPIALLMINMEI.csv")
    ea_cpi = splice_back(ea_cpi, de_cpi)
    world_cpi = chain_weighted([(us_cpi, 0.2 / 0.7), (ea_cpi, 0.5 / 0.7)])

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
    return df


def main() -> None:
    download_all()
    df = build()
    out = df.copy()
    out.insert(0, "quarter", out.index.astype(str))
    out.to_csv(OUT, index=False, float_format="%.6f")
    print(f"\nWrote {OUT}: {len(df)} rows, {df.index[0]}–{df.index[-1]}")
    print(df.head(3))
    print(df.tail(3))


if __name__ == "__main__":
    sys.exit(main())

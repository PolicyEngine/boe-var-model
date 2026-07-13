# Data for the BoE SVAR replication

Build with `python scripts/download_data.py` (conda env `python313`). Raw source
files are cached in `data/raw/`; delete a file to force a fresh download. The
final dataset is `data/boe_var_data.csv`: one row per quarter (`quarter` column,
e.g. `1992Q1`), variables already transformed — **100·log of the level for every
column except `bank_rate`, which is the level in percent**.

Coverage: 1992Q1–2024Q2, 130 quarters, no NaNs (the estimation window
1992Q1–2023Q2 is complete). The binding start is FRED's Brent series
`POILBREUSDQ`, which begins 1992Q1.

## Series and sources

| Column | Description | Source | Series code(s) |
|---|---|---|---|
| `uk_gdp` | UK real GDP, chained volume, SA, £m | ONS (UKEA) | `ABMI` — https://www.ons.gov.uk/generator?format=csv&uri=/economy/grossdomesticproductgdp/timeseries/abmi/ukea |
| `cpisa` | UK CPI, seasonally adjusted (see note) | ONS (MM23) | `D7BT` (CPI all items, NSA, 2015=100) — https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/d7bt/mm23 |
| `cpi_energy` | UK CPI energy (see note) | ONS (MM23) | `D7CH` (CPI 04.5 electricity, gas & other fuels), `D7EC` (CPI 07.2.2 fuels & lubricants) |
| `bank_rate` | Bank Rate, quarterly average, % | BoE IADB | `IUQABEDR` — https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/1988&Dateto=01/Oct/2024&SeriesCodes=IUQABEDR,XUQABK67,XUQAUSS&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N |
| `eri` | Sterling broad effective exchange-rate index, quarterly avg | BoE IADB | `XUQABK67` (same URL) |
| `oil_price` | Real Brent price in sterling (see note) | FRED + BoE | `POILBREUSDQ` (Brent, USD/bbl, IMF via FRED) — https://fred.stlouisfed.org/graph/fredgraph.csv?id=POILBREUSDQ ; `XUQAUSS` (USD per GBP, quarterly avg, BoE) |
| `world_gdp` | World real GDP proxy (see note) | FRED | `GDPC1` (US real GDP), `CLVMNACSCAB1GQEA19` (Euro Area 19 real GDP, 1995Q1–), `CLVMNACSCAB1GQDE` (Germany real GDP, 1991Q1–, used for backcasting) |
| `world_cpi` | World CPI proxy (see note) | FRED | `CPALTT01USQ661S` (US CPI, SA), `CP0000EZ19M086NEST` (EA19 HICP all items, monthly, Dec 1996–), `DEUCPIALLMINMEI` (Germany CPI, monthly, used for backcasting) |

## Construction notes and caveats

- **UK CPI SA (`cpisa`)**: ONS does not publish a seasonally adjusted CPI (the
  Bank's paper uses an internal SA series). We take monthly `D7BT` (NSA),
  average to quarterly, then apply a simple multiplicative classical seasonal
  adjustment (ratio-to-2×4-centered-moving-average, constant quarterly factors
  normalised to mean 1). No X-13; quarterly CPI seasonality is small, but this
  is an approximation to the Bank's CPISA.
- **UK CPI energy (`cpi_energy`)**: no single published CPI "energy" special
  aggregate is downloadable via the ONS API, so we combine CPI division 04.5
  (electricity, gas and other fuels, `D7CH`) and class 07.2.2 (fuels and
  lubricants, `D7EC`) with equal (0.5/0.5) weights via chained weighted log
  growth (base 100 at the first common quarter). CPI weights of the two groups
  are of similar magnitude historically; the exact ONS/BoE energy aggregate
  uses time-varying expenditure weights. Series is NSA.
- **Real oil price (`oil_price`)**: Brent USD/bbl (quarterly average,
  `POILBREUSDQ`) divided by the quarterly-average USD/GBP rate (`XUQAUSS`,
  USD per £1) to get £/bbl, then deflated by our `cpisa` index (×100). Level is
  therefore an index-like real price; only 100·log enters the VAR, so the
  normalisation is irrelevant for the model.
- **Bank Rate**: `IUQABEDR` is the quarterly average of the official Bank Rate.
- **World GDP proxy (`world_gdp`)**: the paper uses the Bank's internal
  UK-trade-weighted world GDP. We proxy it with a chained aggregate of US and
  Euro Area real GDP using rough UK trade shares US 0.2, EA 0.5, other 0.3;
  since no reliable long quarterly series exists for "other", the US/EA weights
  are renormalised to 2/7 and 5/7. EA19 GDP starts 1995Q1 and is backcast to
  1991Q1 using German real GDP growth (`CLVMNACSCAB1GQDE`). The former FRED
  OECD-total series (e.g. `NAEXKP01OEQ652S`) were discontinued when OECD data
  left FRED, so they could not be used. Caveat: excludes China and other EMs,
  so it will differ from the Bank's trade-weighted aggregate, especially
  post-2000.
- **World CPI proxy (`world_cpi`)**: same weighting scheme applied to US CPI
  (SA, `CPALTT01USQ661S`) and EA19 HICP (`CP0000EZ19M086NEST`, monthly NSA,
  averaged to quarterly), the latter backcast before 1997 using German CPI
  (`DEUCPIALLMINMEI`). The EA/DE legs are not seasonally adjusted (HICP
  quarterly seasonality is mild); caveat as for world GDP.
- **Aggregation method**: "chained weighted" means the aggregate index is built
  as 100·exp(cumulative sum of weight-averaged quarterly log growth rates),
  i.e. a Divisia-style fixed-weight chain index. Base = 100 in the first
  quarter where all components exist.
- Sample restricted to 1990Q1–2024Q2 then rows with any NaN dropped; the
  binding start is 1992Q1 (start of FRED `POILBREUSDQ`). The script asserts the
  1992Q1–2023Q2 estimation window is complete.

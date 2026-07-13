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
| `world_gdp` | UK-trade-weighted world real GDP proxy (see note) | FRED | `GDPC1` (US real GDP), `CLVMNACSCAB1GQEA19` (Euro Area 19 real GDP, 1995Q1–), `CLVMNACSCAB1GQDE` (Germany real GDP, used for backcasting), `JPNRGDPEXP` (Japan real GDP, 1994Q1–), `RGDPNACNA666NRUG` (China real GDP at constant national prices, annual, Penn World Table via FRED, interpolated to quarterly) |
| `world_cpi` | UK-trade-weighted world CPI proxy (see note) | FRED | `CPALTT01USQ661S` (US CPI, SA), `CP0000EZ19M086NEST` (EA19 HICP all items, monthly, Dec 1996–), `DEUCPIALLMINMEI` (Germany CPI, used for backcasting), `CPALTT01JPQ661S` (Japan CPI, quarterly, ends 2021Q2, extended with `FPCPITOTLZGJPN` World Bank annual inflation), `CHNCPIALLMINMEI` (China CPI, monthly, 1993M1–) |

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
  UK-trade-weighted world GDP. We proxy it with a chained aggregate of US,
  Euro Area, Japan and China real GDP using approximate UK trade shares
  (goods + services) in three eras, chosen to capture China's rise:

  | Era | EA | US | Japan | China |
  |---|---|---|---|---|
  | 1992Q1–1999Q4 | 0.50 | 0.17 | 0.04 | 0.01 |
  | 2000Q1–2009Q4 | 0.48 | 0.16 | 0.03 | 0.05 |
  | 2010Q1–2024Q2 | 0.45 | 0.16 | 0.02 | 0.07 |

  These are raw shares of total UK trade, informed by ONS Pink Book /
  UK trade statistics and the IMF Direction of Trade Statistics literature
  (EU/euro area ≈45–50% of UK trade throughout; US ≈15–18%; China ≈1% in the
  1990s rising to ≈7% by the 2010s; Japan declining from ≈4% to ≈2%). We
  could not use an official downloadable trade-share series (the ONS Pink
  Book partner tables are not available via the simple time-series API, and
  the OECD/IMF DOTS APIs are not reachable from this build environment), so
  these static era weights are a documented approximation. In each quarter
  the weights are **renormalised over the countries with data** (they cover
  ~0.70–0.72 of UK trade; the remainder — rest of world — is implicitly
  assumed to grow at the weighted average of the included countries).
  Country coverage details: EA19 GDP starts 1995Q1 and is backcast using
  German real GDP growth (`CLVMNACSCAB1GQDE`); Japan GDP (`JPNRGDPEXP`)
  starts 1994Q1, so Japan drops out (weights renormalised) before then;
  China real GDP is the Penn World Table annual series `RGDPNACNA666NRUG`
  (constant national prices), converted to quarterly by log-linear
  interpolation of annual values placed at mid-year — no direct quarterly
  Chinese real GDP is available on FRED since the OECD series
  (`NAEXKP01CNQ189S` etc.) were discontinued. The PWT series ends in 2023,
  so China drops out of the GDP aggregate (weights renormalised over
  US/EA/JP) from 2023Q3 onwards. Interpolation smooths China's quarterly
  profile; its contribution is mainly to trend growth.
- **World CPI proxy (`world_cpi`)**: same era trade weights applied to US CPI
  (SA, `CPALTT01USQ661S`), EA19 HICP (`CP0000EZ19M086NEST`, monthly NSA,
  averaged to quarterly, backcast before 1997 using German CPI
  `DEUCPIALLMINMEI`), Japan CPI (`CPALTT01JPQ661S`, quarterly SA; the OECD
  series ends 2021Q2, after which it is extended with constant within-year
  quarterly growth implied by World Bank annual average inflation
  `FPCPITOTLZGJPN` — an approximation with a flat within-year profile) and
  China CPI (`CHNCPIALLMINMEI`, monthly, starts 1993M1; China drops out of
  the CPI aggregate, weights renormalised, before 1993Q2). The EA/DE legs
  are not seasonally adjusted (HICP quarterly seasonality is mild).
- **Aggregation method**: the aggregate index is built as exp(cumulative sum
  of weight-averaged quarterly log growth rates), i.e. a Divisia-style chain
  index with era-varying weights renormalised each quarter over available
  components, rebased so the 2015 average = 100. Mixed source units/bases are
  therefore irrelevant; only 100·log enters the VAR.
- **v1 comparison file**: `data/boe_var_data_v1_us_ea_only.csv` keeps the
  previous two-country proxies (US 2/7 + EA 5/7 fixed weights, same chain
  method) as `world_gdp`/`world_cpi` in 100·log form for comparison. The new
  aggregates are highly correlated with v1 (levels ≈0.996/1.000, quarterly
  growth ≈0.998/0.989 for GDP/CPI) but the new world GDP grows visibly faster
  over 2000–2010 (China) and the new world CPI slightly slower (Japanese
  deflation, low Chinese inflation).
- Sample restricted to 1990Q1–2024Q2 then rows with any NaN dropped; the
  binding start is 1992Q1 (start of FRED `POILBREUSDQ`). The script asserts the
  1992Q1–2023Q2 estimation window is complete.

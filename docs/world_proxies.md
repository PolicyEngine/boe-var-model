# The world GDP and world CPI proxies

Two of the model's eight variables are not the series the paper uses. This
note says what each proxy is made of, where each piece comes from, why it was
chosen, how far it is from what the Bank publishes about its own series, and
which closer published series exist. `data/README.md` has the build detail;
`scripts/download_data.py` is the build; `tests/test_download_data.py`
(PR #42) checks the arithmetic described here.

Written 27 November 2026. Nothing in the data changes with this note.

## What the paper uses

Brignone and Piffer (2025), Table 1, list "Real World GDP" and "World CPI",
both entering as 100·log, with the note: "World GDP and CPI are UK-trade
weighted." The paper gives no source, country list or weights, and has no
replication package. Figure A-1 plots both series.

What the Bank says elsewhere about its world GDP measure (November 2026
Monetary Policy Report, opened 27 November 2026):

- Chart 1.14, footnote: "UK-weighted world GDP growth is constructed using
  real GDP growth rates of 188 countries weighted according to their shares
  in UK exports." Sources: LSEG Workspace and Bank calculations.
- Table A1.A gives annual growth of "World GDP (UK-weighted)": 3.1% a year
  on average over 1998–2007 and 2.5% over 2010–19. The text gives
  four-quarter growth of 2.3% in 2026Q2.
- The Report's world price variable is "UK-weighted world export prices
  excluding the direct effect of oil prices" (Table 3.B, note h). That is an
  export-price measure, not a consumer-price one. The Report does not
  publish a UK-weighted world CPI.

So the Bank's world GDP has 188 countries and export-share weights; the
proxy has four economies and hand-set weights. For world CPI there is no
published definition to compare with at all.

## The proxies, leg by leg

Both proxies are chained indices: each quarter's log growth is the weighted
average of the legs' log growth, with weights renormalised over the legs
that have data in that quarter, cumulated and rebased to 2015 = 100.

### Weights (both proxies)

| Era | Euro area | United States | Japan | China | Sum |
|---|---|---|---|---|---|
| 1992Q1–1999Q4 | 0.50 | 0.17 | 0.04 | 0.01 | 0.72 |
| 2000Q1–2009Q4 | 0.48 | 0.16 | 0.03 | 0.05 | 0.72 |
| 2010Q1–2026Q2 | 0.45 | 0.16 | 0.02 | 0.07 | 0.70 |

- **Source.** Set by hand as approximate shares of total UK trade in goods
  and services (`TRADE_WEIGHT_ERAS`). They are not taken from a published
  table. `data/README.md` describes them as "informed by" the ONS Pink Book
  and the IMF Direction of Trade Statistics.
- **Why.** The first version used two economies at fixed weights (United
  States 2/7, euro area 5/7). Commit `b14a8e9` (13 July 2026) replaced it
  with four economies and three eras, to bring in China's rising share and
  Japan's falling one. The official partner tables were not available
  through the download routes the build uses.
- **Limits.** Total trade, not exports (the Bank weights by exports). Three
  steps, not annual. The remaining 28–30% of UK trade is assumed to grow
  like the weighted average of the four.

### World GDP (`world_gdp`)

| Leg | Series (all via FRED) | Publisher | Span used | Why this series | Limits |
|---|---|---|---|---|---|
| United States | `GDPC1`, real GDP, quarterly, SA | BEA | 1992Q1– | Official quarterly real GDP | None known |
| Euro area | `CLVMNACSCAB1GQEA19`, EA19 real GDP, quarterly, SA | Eurostat | 1995Q1– | Official euro-area aggregate | Fixed 19-country composition |
| Euro area before 1995 | `CLVMNACSCAB1GQDE`, German real GDP growth | Eurostat | 1992Q1–1994Q4 | The EA19 series starts in 1995Q1; Germany is the largest member | One country stands in for the area, in the years just after reunification |
| Japan | `JPNRGDPEXP`, real GDP, quarterly, SA | Cabinet Office | 1994Q1– | Official quarterly real GDP | Absent before 1994Q2 (weights renormalised) |
| China | `RGDPNACNA666NRUG`, real GDP at constant national prices, **annual** | Penn World Table | to 2023 | No quarterly Chinese real GDP is on FRED since the OECD series stopped | Annual values interpolated log-linearly to quarters, so no quarterly profile; ends 2023, so China is absent from 2023Q3 |

### World CPI (`world_cpi`)

| Leg | Series (all via FRED) | Publisher | Span used | Why this series | Limits |
|---|---|---|---|---|---|
| United States | `CPALTT01USQ661S`, CPI, quarterly, SA | OECD Main Economic Indicators | to 2025Q1 | In the original build | The OECD feed stopped at 2025Q1 |
| United States after 2025Q1 | `CPIAUCSL`, CPI-U, monthly, SA | BLS | 2025Q2– | Official US CPI, still updated | Spliced on growth at 2025Q1 |
| Euro area | `CP0000EZ19M086NEST`, EA19 HICP, monthly, **not seasonally adjusted** | Eurostat | December 1996– | Official euro-area aggregate | Unadjusted; the first "quarter" (1996Q4) is December alone (issue #41) |
| Euro area before 1997 | `DEUCPIALLMINMEI`, German CPI, monthly, **not seasonally adjusted** | OECD Main Economic Indicators | 1992Q1–1996Q4 | HICP starts in December 1996 | One country stands in for the area; unadjusted |
| Japan | `CPALTT01JPQ661S`, CPI, quarterly, SA | OECD Main Economic Indicators | to 2021Q2 | In the original build | The FRED copy's last observation is 2021Q2 |
| Japan after 2021Q2 | `FPCPITOTLZGJPN`, **annual** average inflation | World Bank | 2021Q3–2025Q4 | A way to carry the level forward | Each quarter of a year grows at the same rate; ends 2025, so Japan is absent from 2026Q1 |
| China | `CHNCPIALLMINMEI`, CPI, monthly, not seasonally adjusted | OECD Main Economic Indicators | January 1993–April 2025 | In the original build | Absent before 1993Q2 and from 2025Q3; its 2025Q2 value is April alone (issue #41) |

The aggregate is then seasonally adjusted with time-varying factors
(`seasonal_adjust_tv`), because the euro-area and German legs are unadjusted
and their pattern drifts (`docs/decisions.md`, 18 November 2026). The
unadjusted aggregate is kept in `data/boe_var_data_world_cpi_nsa.csv`.

The FRED pages for the two stale OECD series were opened on 27 November
2026: `CPALTT01JPQ661S` shows a last observation of 2021Q2 and a last update
of 15 May 2024; `CHNCPIALLMINMEI` shows April 2025 and 15 May 2025. Neither
lists a next release date. `JPNCPIALLMINMEI`, the monthly OECD Japanese CPI
on FRED, also stops (June 2021), so it is not a substitute.

## How far the world GDP proxy is from the Bank's published figures

Average annual growth, from `data/boe_var_data.csv` (annual averages of the
level) against Table A1.A of the November 2026 Report:

| | 1998–2007 | 2010–19 | Four-quarter growth, 2026Q2 |
|---|---|---|---|
| Bank, UK-weighted world GDP | 3.1% | 2.5% | 2.3% |
| This proxy | 2.92% | 2.25% | 1.43% |
| First version (United States and euro area only) | 2.59% | 1.72% | |

The four-economy proxy is 0.2 to 0.25 points a year below the Bank's measure
over the two decades the Bank reports, and closer to it than the first
version. At the end of the sample the gap is larger (0.9 points in 2026Q2):
China has no data after 2023Q2 here, and the faster-growing rest of the
world was never in. These are the only numbers of the Bank's series this
note could check; they are averages, and say nothing about how well the
proxy tracks the Bank's series quarter to quarter.

## Why this matters for the impulse responses

PR #37 (open, not merged) compares this replication's impulse responses with
medians read from the paper's Figures 2–3. Of the seven panels where the
paper's median leaves this replication's 90% set, five have a proxied world
series as the response variable: world CPI after the world demand, world
supply, UK supply and UK monetary-policy shocks, and world GDP after the UK
supply shock. That is an association. It has not been shown that the proxies
cause those gaps: the same PR finds that a looser prior closes much of the
distance on the same data. What can be said is that the world block is the
part of the model where the inputs are known to differ from the paper's.

## Closer published series

Assessed on 27 November 2026. Each row says what was opened. None is adopted
here; issue #43 proposes them. The repository's scoring note
(`results/score_2026q3.md`) lists the publishers approved for download as
ONS, the Bank of England, the ECB/Eurostat and Destatis, so rows outside
that list need a decision first.

| Leg | Closer series | What was opened | Approved publisher? | Assessment |
|---|---|---|---|---|
| Euro-area CPI, 1997– | ECB seasonally and working-day adjusted HICP (`HICP.M.U2.Y.000000.4F0.INX`) | Already in `data/external/` (retrieved 19 November); `results/ecb_seasonal_check.md` | Yes | Closer: adjusted at source, so the home-made filter would no longer carry the euro-area leg. It removes nearly the same pattern as the filter (correlation 0.95, `docs/decisions.md`). Deferred there to the next data refresh. |
| Japan CPI, 2021Q3– | Statistics Bureau of Japan monthly CPI | https://www.stat.go.jp/english/data/cpi/index.html (monthly results, 2025 base, current) | No | Closer: a monthly official index in place of an annual rate spread evenly over four quarters, and it does not stop in 2025. The page opened does not show a seasonally adjusted index or a stable download address; both need checking. |
| Weights | Bank of England sterling ERI weights, annual | https://www.bankofengland.co.uk/statistics/sterling-exchange-rate-index (narrow index, 2024: euro area 40.2%, United States 22.0%, China 10.2%; spreadsheet linked) | Yes | Published and annual, but a different concept: competitiveness weights for trade in manufactures and services, including third-market competition, not export shares. Rescaled to the four economies the era weights are 64% / 23% / 10% for the euro area, United States and China; the ERI's relative weight on the United States is higher. Not clearly closer to the Bank's export-share weights. ONS trade by partner country would be; it was not opened. |
| World GDP, whole series | The Bank's own UK-weighted world GDP | November 2026 Report: Chart 1.14 (four-quarter growth, recent quarters), Table A1.A (annual) | Yes | The Bank publishes growth rates for recent quarters and annual figures, with chart data in spreadsheets. No quarterly level series from 1992 was found, and the chart-data files were not opened (the download page address tried returned 404). Usable as a check on the proxy, as in the table above, not as a replacement. |
| World GDP and CPI, whole series | Dallas Fed Database of Global Economic Indicators | https://www.dallasfed.org/research/international/dgei (search result and page summary) | No | Not closer: its aggregates exclude the United States and were built on US trade weights. |
| Euro area before 1995 (GDP) and 1997 (CPI) | Area-Wide Model database (euro-area aggregates from 1970) | https://eabcn.org/node/19 (search result: historical database to 2017Q4, not updated since 2019) | No (hosted by EABCN; built at the ECB) | Closer in concept: an area aggregate in place of Germany alone for 1992–1996. The series definitions were not opened. |
| China GDP | A quarterly official series | Not opened | No | Not assessed. The annual interpolation and the 2023 end are the weakest part of the GDP proxy at the sample end. |
| World CPI, whole series | None found | November 2026 Report | | The Bank publishes no UK-weighted world CPI. |

## What would change if a leg were replaced

Every artefact is built on `data/boe_var_data.csv`: the replication summary
and figures, the forecasts, the evaluation JSONs and the stored comparisons.
Replacing a leg regenerates all of them. That is why this note proposes and
does not change.

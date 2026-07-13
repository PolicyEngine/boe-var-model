# Replication summary

- Posterior draws: 2000; accepted identified draws: 500; lags: 4.
- Sample: 1992Q1–2023Q2 (126 quarters).

## FEVD at 1-year horizon (median shares)

| Variable | Identified global | Identified domestic | Unidentified |
|---|---|---|---|
| UK GDP | 43.4% | 35.6% | 21.0% |
| UK CPI | 44.1% | 34.7% | 21.2% |

Paper benchmark: identified global shocks explain roughly ~40% of UK GDP and ~50% of UK CPI variation at business-cycle horizons.

Known discrepancy: the paper reports UK monetary policy as the largest domestic contributor to CPI variance; in this replication it is not (see detailed table) — likely driven by the proxy world aggregates and fewer accepted draws.

## Detailed 1-year FEVD (median, %)

| Variable | World demand | World energy | World supply | Unident. global | UK demand | UK supply | UK mon. pol. | Unident. UK |
|---|---|---|---|---|---|---|---|---|
| World GDP | 32.4 | 28.2 | 21.5 | 16.9 | 0.2 | 0.2 | 0.2 | 0.2 |
| World CPI | 50.0 | 20.1 | 7.3 | 21.7 | 0.2 | 0.3 | 0.2 | 0.2 |
| Oil price | 21.1 | 38.1 | 23.9 | 15.8 | 0.4 | 0.2 | 0.2 | 0.3 |
| Bank Rate | 30.7 | 4.0 | 4.8 | 14.0 | 20.5 | 8.8 | 8.1 | 9.1 |
| Exch. rate | 7.5 | 12.8 | 5.0 | 7.9 | 16.5 | 15.9 | 15.3 | 19.1 |
| UK CPISA | 25.7 | 15.9 | 2.5 | 11.2 | 10.9 | 16.6 | 7.2 | 9.9 |
| CPI Energy | 33.3 | 27.2 | 0.2 | 12.5 | 5.7 | 7.9 | 5.3 | 8.0 |
| UK GDP | 18.3 | 15.4 | 9.7 | 8.8 | 10.7 | 17.1 | 7.8 | 12.2 |

Figures: fig2_irf_world.png, fig3_irf_uk.png, fig4_fevd.png, fig5_shocks.png, fig6_hist_decomp.png.

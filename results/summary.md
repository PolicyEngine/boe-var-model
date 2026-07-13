# Replication summary

- Posterior draws: 3000; accepted identified draws: 186 (acceptance rate 6.2%); lags: 4.
- Importance-weight effective sample size (ESS): 91.0.
- Sample: 1992Q1–2023Q2 (126 quarters).

## FEVD at 1-year horizon (median shares)

| Variable | Identified global | Identified domestic | Unidentified |
|---|---|---|---|
| UK GDP | 42.5% | 36.7% | 20.8% |
| UK CPI | 52.2% | 30.1% | 17.7% |

Paper benchmark: identified global shocks explain roughly ~40% of UK GDP and ~50% of UK CPI variation at business-cycle horizons.

Known discrepancy: the paper reports UK monetary policy as the largest domestic contributor to CPI variance; in this replication it is not (see detailed table) — likely driven by the proxy world aggregates and fewer accepted draws.

## Detailed 1-year FEVD (median, %)

| Variable | World demand | World energy | World supply | Unident. global | UK demand | UK supply | UK mon. pol. | Unident. UK |
|---|---|---|---|---|---|---|---|---|
| World GDP | 39.6 | 30.7 | 21.1 | 7.6 | 0.3 | 0.2 | 0.2 | 0.2 |
| World CPI | 58.4 | 23.0 | 6.2 | 11.6 | 0.2 | 0.2 | 0.2 | 0.1 |
| Oil price | 15.7 | 41.1 | 32.0 | 10.2 | 0.4 | 0.2 | 0.2 | 0.2 |
| Bank Rate | 34.2 | 4.3 | 3.9 | 17.1 | 20.2 | 6.3 | 5.7 | 8.5 |
| Exch. rate | 5.6 | 9.4 | 2.8 | 4.7 | 17.1 | 18.0 | 19.3 | 23.0 |
| UK CPISA | 31.2 | 18.8 | 2.2 | 4.9 | 7.1 | 15.6 | 7.4 | 12.8 |
| CPI Energy | 32.0 | 25.8 | 0.2 | 9.7 | 7.7 | 8.5 | 6.7 | 9.2 |
| UK GDP | 19.3 | 13.3 | 9.8 | 8.9 | 7.0 | 23.1 | 6.6 | 11.9 |

Figures: fig2_irf_world.png, fig3_irf_uk.png, fig4_fevd.png, fig5_shocks.png, fig6_hist_decomp.png.

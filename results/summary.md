# Replication summary

- Posterior draws: 10000; accepted identified draws: 751 (acceptance rate 7.5%); lags: 4.
- Importance-weight effective sample size (ESS): 350.3.
- Sample: 1992Q1–2023Q2 (126 quarters).
- Hyperparameters: lam = 0.2000, mu = 1.0000, theta = 1.0000.

## FEVD at 1-year horizon (median shares)

| Variable | Identified global | Identified domestic | Unidentified |
|---|---|---|---|
| UK GDP | 40.9% | 38.0% | 21.1% |
| UK CPI | 50.1% | 35.4% | 14.5% |

Paper benchmark: identified global shocks explain roughly ~40% of UK GDP and ~50% of UK CPI variation at business-cycle horizons.

Known discrepancy: the paper reports UK monetary policy as the largest domestic contributor to CPI variance; in this replication it is not (see detailed table) — likely driven by the proxy world aggregates and fewer accepted draws.

## Detailed 1-year FEVD (median, %)

| Variable | World demand | World energy | World supply | Unident. global | UK demand | UK supply | UK mon. pol. | Unident. UK |
|---|---|---|---|---|---|---|---|---|
| World GDP | 39.9 | 26.0 | 19.4 | 13.8 | 0.2 | 0.2 | 0.2 | 0.3 |
| World CPI | 53.5 | 25.9 | 7.1 | 12.7 | 0.2 | 0.3 | 0.2 | 0.2 |
| Oil price | 15.9 | 40.2 | 33.2 | 9.7 | 0.4 | 0.3 | 0.1 | 0.2 |
| Bank Rate | 34.7 | 5.2 | 4.7 | 13.1 | 19.3 | 7.9 | 5.9 | 9.2 |
| Exch. rate | 6.8 | 8.1 | 3.5 | 7.8 | 17.1 | 14.9 | 19.8 | 21.9 |
| UK CPISA | 28.2 | 18.3 | 3.6 | 7.0 | 11.2 | 16.2 | 8.0 | 7.5 |
| CPI Energy | 34.6 | 27.1 | 0.2 | 7.6 | 6.3 | 10.3 | 6.5 | 7.5 |
| UK GDP | 20.1 | 13.6 | 7.2 | 8.9 | 7.7 | 21.9 | 8.4 | 12.2 |

Figures: fig2_irf_world.png, fig3_irf_uk.png, fig4_fevd.png, fig5_shocks.png, fig6_hist_decomp.png.

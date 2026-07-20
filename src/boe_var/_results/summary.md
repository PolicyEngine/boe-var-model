# Replication summary

- Posterior draws: 10000; accepted identified draws: 751 (acceptance rate 7.5%); lags: 4.
- Importance-weight effective sample size (ESS): 355.9.
- Sample: 1992Q1–2025Q1 (133 quarters).
- Hyperparameters: lam = 0.2000, mu = 1.0000, theta = 1.0000.

## FEVD at 1-year horizon (median shares)

| Variable | Identified global | Identified domestic | Unidentified |
|---|---|---|---|
| UK GDP | 42.1% | 37.8% | 20.1% |
| UK CPI | 49.5% | 35.9% | 14.6% |

Paper benchmark: identified global shocks explain roughly ~40% of UK GDP and ~50% of UK CPI variation at business-cycle horizons.

Known discrepancy: the paper reports UK monetary policy as the largest domestic contributor to CPI variance; in this replication it is not (see detailed table) — likely driven by the proxy world aggregates and fewer accepted draws.

## Detailed 1-year FEVD (median, %)

| Variable | World demand | World energy | World supply | Unident. global | UK demand | UK supply | UK mon. pol. | Unident. UK |
|---|---|---|---|---|---|---|---|---|
| World GDP | 38.3 | 26.7 | 20.8 | 13.2 | 0.2 | 0.3 | 0.2 | 0.3 |
| World CPI | 52.0 | 25.2 | 6.0 | 16.0 | 0.2 | 0.2 | 0.2 | 0.2 |
| Oil price | 14.9 | 37.6 | 34.7 | 12.1 | 0.3 | 0.2 | 0.1 | 0.2 |
| Bank Rate | 33.1 | 4.9 | 5.0 | 11.8 | 19.9 | 9.1 | 6.3 | 9.8 |
| Exch. rate | 6.2 | 7.8 | 3.0 | 7.4 | 18.5 | 16.1 | 21.1 | 19.9 |
| UK CPISA | 27.1 | 20.0 | 2.5 | 7.4 | 12.4 | 13.5 | 10.1 | 7.2 |
| CPI Energy | 27.8 | 24.7 | 0.2 | 9.1 | 6.3 | 11.8 | 9.6 | 10.5 |
| UK GDP | 20.5 | 14.5 | 7.2 | 7.6 | 8.9 | 20.0 | 8.9 | 12.5 |

Figures: fig2_irf_world.png, fig3_irf_uk.png, fig4_fevd.png, fig5_shocks.png, fig6_hist_decomp.png.

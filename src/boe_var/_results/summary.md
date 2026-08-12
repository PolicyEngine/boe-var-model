# Replication summary

- Posterior draws: 10000; accepted identified draws: 744 (acceptance rate 7.4%); lags: 4; seed: 20260812.
- Importance-weight effective sample size (ESS): 356.1.
- Sample: 1992Q1–2025Q1 (133 quarters).
- Hyperparameters: lam = 0.2000, mu = 1.0000, theta = 1.0000.

## FEVD at the 1-year horizon (4-quarter-ahead forecast error)

Sum of the per-shock posterior medians, renormalised to 100%. This is the historical presentation and is kept for continuity, but the median of a sum is not the sum of medians -- see the posterior table below for the group share formed on each draw.

| Variable | Identified global | Identified domestic | Unidentified |
|---|---|---|---|
| UK GDP | 42.8% | 37.7% | 19.5% |
| UK CPI | 49.5% | 34.0% | 16.5% |

### Posterior of the group share (formed per draw)

The same 4-quarter-ahead shares, but summed over each group **on every accepted draw** before quantiles are taken. This is the quantity the ~40% / ~50% claim is about and it comes with a band; the point estimates differ from the table above because the median of a sum is not the sum of medians.

The **mean** column is the paper-comparable statistic: the paper's Figure 4 plots the *mean* of the decompositions and its note states the gap to 100% is the unidentified shocks, so its shares are of TOTAL forecast-error variance and are not renormalised over the identified shocks.

| Variable | Group | Mean | Median | 68% band | 90% band |
|---|---|---|---|---|---|
| UK GDP | global | 37.4% | 36.1% | [23.4%, 52.2%] | [14.4%, 63.6%] |
| UK GDP | domestic | 38.9% | 40.1% | [18.9%, 56.3%] | [11.5%, 64.8%] |
| UK GDP | unidentified | 23.7% | 21.6% | [6.2%, 40.5%] | [1.5%, 53.9%] |
| UK CPI | global | 42.3% | 42.0% | [25.5%, 60.3%] | [14.0%, 75.5%] |
| UK CPI | domestic | 36.6% | 37.3% | [17.1%, 53.8%] | [8.5%, 61.2%] |
| UK CPI | unidentified | 21.1% | 17.5% | [5.9%, 37.0%] | [2.0%, 50.5%] |

Paper benchmark: identified global shocks explain roughly ~40% of UK GDP and ~50% of UK CPI variation one year after the shocks. Compare against the **mean** column above, not the renormalised table: on the paper's own definition this replication matches on UK GDP and falls materially short on UK CPI. The renormalised table closes that gap arithmetically rather than economically.

Note on the domestic ranking: the paper's prose says UK monetary policy is the largest domestic contributor, but it publishes no FEVD table and its Figure 4 does not obviously show that for CPI. This replication finds UK demand and UK supply larger than monetary policy for CPI. Treat as unresolved rather than as a known defect; proxy world aggregates remain a candidate explanation.

## Detailed 1-year FEVD (median, %)

| Variable | World demand | World energy | World supply | Unident. global | UK demand | UK supply | UK mon. pol. | Unident. UK |
|---|---|---|---|---|---|---|---|---|
| World GDP | 38.7 | 24.0 | 22.0 | 14.4 | 0.2 | 0.2 | 0.2 | 0.2 |
| World CPI | 59.9 | 21.7 | 5.9 | 11.7 | 0.2 | 0.2 | 0.2 | 0.2 |
| Oil price | 19.4 | 37.6 | 33.3 | 9.1 | 0.2 | 0.1 | 0.1 | 0.1 |
| Bank Rate | 30.7 | 4.4 | 5.7 | 13.4 | 22.8 | 7.4 | 6.8 | 8.9 |
| Exch. rate | 5.0 | 8.2 | 3.0 | 6.2 | 18.6 | 17.5 | 18.9 | 22.5 |
| UK CPISA | 27.2 | 19.4 | 2.9 | 5.8 | 11.4 | 13.9 | 8.7 | 10.8 |
| CPI Energy | 30.7 | 24.0 | 0.1 | 7.3 | 6.5 | 12.3 | 8.2 | 10.8 |
| UK GDP | 19.9 | 13.0 | 9.8 | 9.5 | 9.5 | 19.1 | 9.1 | 10.0 |

Figures: fig2_irf_world.png, fig3_irf_uk.png, fig4_fevd.png, fig5_shocks.png, fig6_hist_decomp.png.

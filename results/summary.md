# Replication summary

- Posterior draws: 10000; accepted identified draws: 765 (acceptance rate 7.6%); lags: 4; seed: 20260812.
- Importance-weight effective sample size (ESS): 368.0.
- Sample: 1992Q1–2025Q1 (133 quarters).
- Hyperparameters: lam = 0.2000, mu = 1.0000, theta = 1.0000.

## FEVD at the 1-year horizon (4-quarter-ahead forecast error)

Sum of the per-shock posterior medians, renormalised to 100%. This is the historical presentation and is kept for continuity, but the median of a sum is not the sum of medians -- see the posterior table below for the group share formed on each draw.

| Variable | Identified global | Identified domestic | Unidentified |
|---|---|---|---|
| UK GDP | 43.2% | 36.5% | 20.3% |
| UK CPI | 48.0% | 34.0% | 18.0% |

### Posterior of the group share (formed per draw)

The same 4-quarter-ahead shares, but summed over each group **on every accepted draw** before quantiles are taken. This is the quantity the ~40% / ~50% claim is about and it comes with a band; the point estimates differ from the table above because the median of a sum is not the sum of medians.

The **mean** column is the paper-comparable statistic: the paper's Figure 4 plots the *mean* of the decompositions and its note states the gap to 100% is the unidentified shocks, so its shares are of TOTAL forecast-error variance and are not renormalised over the identified shocks.

| Variable | Group | Mean | Median | 68% band | 90% band |
|---|---|---|---|---|---|
| UK GDP | global | 38.7% | 37.5% | [24.0%, 53.1%] | [15.7%, 65.0%] |
| UK GDP | domestic | 38.6% | 40.0% | [19.8%, 56.2%] | [10.6%, 63.0%] |
| UK GDP | unidentified | 22.7% | 19.2% | [5.5%, 39.9%] | [1.8%, 52.6%] |
| UK CPI | global | 41.5% | 41.9% | [25.6%, 56.3%] | [14.5%, 70.5%] |
| UK CPI | domestic | 36.8% | 37.4% | [19.9%, 53.9%] | [8.8%, 59.5%] |
| UK CPI | unidentified | 21.7% | 17.8% | [5.4%, 38.5%] | [1.9%, 54.5%] |

Paper benchmark: identified global shocks explain roughly ~40% of UK GDP and ~50% of UK CPI variation one year after the shocks. Compare against the **mean** column above, not the renormalised table: on the paper's own definition this replication matches on UK GDP and falls materially short on UK CPI. The renormalised table closes that gap arithmetically rather than economically.

Note on the domestic ranking: the paper's prose says UK monetary policy is the largest domestic contributor, but it publishes no FEVD table and its Figure 4 does not obviously show that for CPI. This replication finds UK demand and UK supply larger than monetary policy for CPI. Treat as unresolved rather than as a known defect; proxy world aggregates remain a candidate explanation.

## Detailed 1-year FEVD (median, %)

| Variable | World demand | World energy | World supply | Unident. global | UK demand | UK supply | UK mon. pol. | Unident. UK |
|---|---|---|---|---|---|---|---|---|
| World GDP | 39.5 | 26.9 | 19.6 | 13.1 | 0.2 | 0.2 | 0.2 | 0.3 |
| World CPI | 59.1 | 24.1 | 5.2 | 10.9 | 0.2 | 0.2 | 0.2 | 0.2 |
| Oil price | 17.4 | 34.0 | 35.7 | 12.3 | 0.3 | 0.2 | 0.1 | 0.1 |
| Bank Rate | 32.0 | 4.7 | 5.4 | 12.8 | 19.2 | 8.2 | 6.9 | 10.8 |
| Exch. rate | 5.2 | 7.7 | 2.9 | 6.9 | 19.0 | 12.7 | 20.6 | 25.0 |
| UK CPISA | 27.1 | 18.1 | 2.8 | 6.3 | 10.5 | 14.6 | 8.9 | 11.7 |
| CPI Energy | 30.4 | 23.1 | 0.1 | 8.1 | 6.9 | 11.0 | 11.0 | 9.4 |
| UK GDP | 21.1 | 12.7 | 9.4 | 9.1 | 9.9 | 18.2 | 8.4 | 11.3 |

Figures: fig2_irf_world.png, fig3_irf_uk.png, fig4_fevd.png, fig5_shocks.png, fig6_hist_decomp.png.

# Replication summary

- Posterior draws: 10000; accepted identified draws: 731 (acceptance rate 7.3%); lags: 4; seed: 20260812.
- Importance-weight effective sample size (ESS): 333.3.
- Sample: 1992Q1–2025Q1 (133 quarters).
- Hyperparameters: lam = 0.2000, mu = 1.0000, theta = 1.0000.

## FEVD at the 1-year horizon (4-quarter-ahead forecast error)

Sum of the per-shock posterior medians, renormalised to 100%. This is the historical presentation and is kept for continuity, but the median of a sum is not the sum of medians -- see the posterior table below for the group share formed on each draw.

| Variable | Identified global | Identified domestic | Unidentified |
|---|---|---|---|
| UK GDP | 42.9% | 35.0% | 22.1% |
| UK CPI | 59.0% | 19.9% | 21.2% |

### Posterior of the group share (formed per draw)

The same 4-quarter-ahead shares, but summed over each group **on every accepted draw** before quantiles are taken. This is the quantity the ~40% / ~50% claim is about and it comes with a band; the point estimates differ from the table above because the median of a sum is not the sum of medians.

The **mean** column is the paper-comparable statistic: the paper's Figure 4 plots the *mean* of the decompositions and its note states the gap to 100% is the unidentified shocks, so its shares are of TOTAL forecast-error variance and are not renormalised over the identified shocks.

| Variable | Group | Mean | Median | 68% band | 90% band |
|---|---|---|---|---|---|
| UK GDP | global | 37.6% | 37.4% | [23.2%, 50.3%] | [14.6%, 63.5%] |
| UK GDP | domestic | 39.0% | 40.8% | [20.3%, 55.1%] | [8.8%, 61.7%] |
| UK GDP | unidentified | 23.4% | 19.7% | [5.9%, 39.9%] | [2.0%, 60.9%] |
| UK CPI | global | 52.0% | 52.6% | [34.3%, 67.9%] | [23.9%, 80.3%] |
| UK CPI | domestic | 24.8% | 24.6% | [10.9%, 37.3%] | [4.7%, 44.9%] |
| UK CPI | unidentified | 23.2% | 19.8% | [6.7%, 41.7%] | [2.1%, 57.1%] |

Paper benchmark: identified global shocks explain roughly ~40% of UK GDP and ~50% of UK CPI variation one year after the shocks. Compare against the **mean** column above, not the renormalised table: on the paper's own definition this replication matches on UK GDP and falls materially short on UK CPI. The renormalised table closes that gap arithmetically rather than economically.

Note on the domestic ranking: the paper's prose says UK monetary policy is the largest domestic contributor, and it publishes no FEVD table. Read approximately off its Figure 4 at one year, monetary policy explains about 20% of UK GDP (UK supply about 10%, UK demand about 9%) and about 9% of UK CPI (UK demand about 14%, UK supply about 5%). Here (posterior means, table below) monetary policy explains 11.2% of UK GDP (UK supply 14.9%, UK demand 12.9%) and 7.2% of UK CPI (UK demand 6.4%, UK supply 11.3%). The clearest open discrepancy is the monetary-policy share of UK GDP, well below the paper's; the read-offs are visual, so treat this as unresolved rather than as a known defect.

## Detailed 1-year FEVD (posterior mean, %)

Per-draw shares averaged over the accepted draws (importance-weighted). Rows sum to 100 without rescaling; this is the statistic the paper's Figure 4 plots.

| Variable | World demand | World energy | World supply | Unident. global | UK demand | UK supply | UK mon. pol. | Unident. UK |
|---|---|---|---|---|---|---|---|---|
| World GDP | 35.0 | 23.6 | 23.6 | 16.7 | 0.3 | 0.3 | 0.3 | 0.4 |
| World CPI | 48.2 | 24.2 | 6.3 | 20.2 | 0.2 | 0.3 | 0.3 | 0.3 |
| Oil price | 20.7 | 33.2 | 27.0 | 18.1 | 0.3 | 0.2 | 0.2 | 0.3 |
| Bank Rate | 25.1 | 7.4 | 6.3 | 14.0 | 16.4 | 10.6 | 9.6 | 10.5 |
| Exch. rate | 5.7 | 8.8 | 4.3 | 7.6 | 18.5 | 17.7 | 18.7 | 18.8 |
| UK CPISA | 26.1 | 18.2 | 7.6 | 13.2 | 6.4 | 11.3 | 7.2 | 10.0 |
| CPI Energy | 22.9 | 19.7 | 0.3 | 11.4 | 8.4 | 12.8 | 11.6 | 12.9 |
| UK GDP | 18.2 | 11.7 | 7.7 | 9.9 | 12.9 | 14.9 | 11.2 | 13.6 |

Figures: fig2_irf_world.png, fig3_irf_uk.png, fig4_fevd.png, fig5_shocks.png, fig6_hist_decomp.png.

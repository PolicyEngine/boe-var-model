# Decisions

Dated modelling decisions, the evidence behind each, and what was left out.
Newest first. `docs/band_decision.md` is the longer note for the 90% band
(issue #12); `docs/improvements.md` tracks the toolbox-review items.

## 25 November 2026 (Fable 5.2 validity round; issue #11)

### Impulse responses against the paper's Figures 2–3: measured, and one open decision

**What was done.** The paper publishes its impulse responses only as
figures. Their medians were read from the vector drawing in the PDF
(`scripts/extract_paper_irf_medians.py`,
`data/external/brignone_piffer_2025_fig2_fig3_irf_medians.csv`; method,
checks and limits in `data/external/README.md`). The published run's
impulse responses are pinned in `results/irf_reference.json` and compared
in `results/irf_vs_paper.md`. The gates are in
`tests/test_irf_benchmark.py`.

**Provenance is a decision for Vahid.** Issue #11 says reference numbers
must not be read off figure pixels. These were not read off pixels: the
medians are path coordinates in the file, good to half a 0.12 pt grid step.
But the y-axis tick values are transcribed by hand from a rendering, and
the authors have not confirmed anything. The transcription is checked four
ways against the drawing and three ways against the paper (Table 2 zeros,
Table 2 signs, the prose of section 3.1), and all pass. If that is not
acceptable as a reference, delete the CSV and the `paper_comparison` block;
the pinned `irf` block and its tests stand on their own.

**What the comparison shows** (published run, 731 accepted draws):

- On impact the two agree. For 22 of the 23 responses Table 2 restricts in
  sign, this replication's median is 0.94–1.57 times the paper's (median
  1.21). The exception is world supply → UK CPI (−0.13% here, −0.04% in the
  paper).
- The paper's median is inside this replication's 90% set at 93% of 1,008
  points, inside the 68% set at 76%, and the medians share a sign at 84%.
- Seven of 48 panels have the paper's median outside the 90% set at some
  quarter. Five involve world CPI, which is a proxy here.
- The largest economic difference is persistence. After a monetary-policy
  shock the paper's Bank Rate response is 8bp on impact and 1bp after five
  years; here it is 9bp and 5bp. The paper's UK CPI response is −0.42%
  after five years; here −0.20%.

**What explains the gap.** The same run with one setting changed at a time
(10,000 draws, seed 20260812; GitHub Actions run 45517819776, macOS leg;
`scripts/run_irf_reference.py --est-end 2023Q2`, `--lam 1.04`, `--lags 2`,
each with `--out-json` and `--out-md` pointing away from `results/`):

| Run | Accepted / ESS | Paper median inside 68% / 90% set | Same sign | Distance between medians, median / mean panel | Impact ratio (median) | MP → Bank Rate, h0 / h20 (bp) | MP → UK CPI, h20 (%) | World supply → UK GDP, h20 (%) |
|---|---|---|---|---|---|---|---|---|
| Paper (read from the figures) | 10,000 accepted | | | | | 8.1 / 0.8 | −0.42 | 0.13 |
| Published: to 2025Q1, λ 0.2, 4 lags | 731 / 333 | 76% / 93% | 84% | 33% / 46% | 1.21 | 9.1 / 5.0 | −0.20 | −0.11 |
| Sample to 2023Q2 (the paper's) | 709 / 349 | 80% / 95% | 83% | 36% / 45% | 1.25 | 9.3 / 5.2 | −0.18 | −0.10 |
| λ = 1.04 (marginal-likelihood optimum) | 607 / 251 | 98% / 100% | 90% | 25% / 35% | 1.05 | 6.5 / 1.1 | −0.35 | 0.01 |
| 2 lags | 763 / 371 | 73% / 88% | 82% | 38% / 48% | 1.28 | 12.5 / 5.7 | −0.18 | −0.12 |

"Distance between medians" is the mean absolute difference over the 21
quarters as a share of the paper's largest response in the panel. Unlike
the "inside" shares it does not improve just because a looser prior widens
the sets.

- The longer sample is not the cause, and nor is the lag length.
- The prior tightness is the one setting tried that moves the replication
  towards the paper, and it does so on every measure: sign agreement, the
  distance between medians, impact magnitudes, and the persistence of the
  monetary-policy responses.
- This fits what the paper says about its prior. It follows Giannone,
  Lenza and Primiceri (2015), whose method chooses the tightness from the
  data. λ = 0.2 is a conventional fixed value. The marginal likelihood also
  prefers λ ≈ 1.04 by 68 log points (issue #11, explosive-roots comment).
- It does not close the gap entirely, and the world series are still
  proxies.

**Decision needed (not taken here).** Whether to move the published
replication, and the forecasts built on the same prior, from λ = 0.2 to the
marginal-likelihood optimum. For: closer to the paper's method and to its
figures, and a higher marginal likelihood. Against: it changes every
published impulse response, FEVD share and forecast; fewer draws are
accepted (607 against 731); and its effect on forecast accuracy and band
coverage has not been measured (the coverage and rolling evaluations would
have to be re-run at the new λ first). Nothing published changes in this
round.

## 18 November 2026 (Fable 5.2 review round; PR #34)

### 1. World CPI: seasonally adjust the proxy (adopted)

**Decision.** Adopt the time-varying seasonal adjustment of the world CPI
proxy (`seasonal_adjust_tv`, 11-year window) and regenerate every artefact
on it.

**Why.** The proxy averages euro-area HICP and German CPI, which are not
seasonally adjusted at source. Quarterly world CPI growth then has a
seasonal pattern that the VAR does not absorb, and the pattern drifts, so
constant factors do not remove it. One-way F test of quarterly log growth
across the four quarters, by era (recomputed 18 Nov from
`data/boe_var_data_world_cpi_nsa.csv` and `data/boe_var_data.csv`):

| Era | Unadjusted: F (p) | Adjusted: F (p) | Unadjusted mean growth Q1 / Q2 / Q3 / Q4 |
|---|---|---|---|
| 1993Q1–2004Q4 | 5.09 (0.004) | 0.38 (0.77) | 0.76 / 0.62 / 0.33 / 0.39 |
| 2005Q1–2014Q4 | 6.82 (0.001) | 0.03 (0.99) | 0.30 / 0.94 / 0.18 / 0.49 |
| 2015Q1–2021Q4 | 6.54 (0.002) | 0.39 (0.76) | −0.05 / 0.96 / 0.18 / 0.51 |
| 2022Q1–2025Q1 | 0.90 (0.48) | 0.06 (0.98) | 0.84 / 1.66 / 0.79 / 0.77 |
| 1993Q1–2025Q1 | 9.21 (<0.001) | 0.02 (1.00) | 0.46 / 0.89 / 0.29 / 0.48 |

The same test on the VAR's world CPI residuals fails on the unadjusted data
and passes in every era on the adjusted data
(`tests/test_data_seasonality.py`). The adjustment changes year-on-year
world CPI inflation by at most 0.09pp, so it removes the within-year
pattern and leaves the signal.

**The criterion was the seasonality test, not closeness to the paper.** The
paper's FEVD cannot settle it, because one share moves towards the paper
and one moves away:

| UK CPI forecast-error variance, 1 year ahead | Unadjusted | Adjusted | Paper (about) |
|---|---|---|---|
| Identified global shocks | 41.5% | 52.0% | 50% |
| Identified domestic shocks | 36.8% | 24.8% | 40% |
| Unidentified | 21.7% | 23.2% | 20% |

The paper's three round figures (section 3: global about 50% of UK CPI,
domestic about 40%, identified shocks about 80% of the variance of most
domestic variables) do not add up exactly, so they are a rough guide, not
a target.

**What it changes.** Replication: 765 accepted draws (ESS 368) become 731
(ESS 333). The reading of the latest quarter (2026Q2) changes for one
shock: the probability that the world supply shock was positive moves from
0.28 to 0.55, so "likely a negative world supply shock" becomes "no call".
Fan medians move by less than seed-to-seed noise (0.1–0.17pp), except CPI
in 2026Q3 (about −0.04pp). Mean 90% coverage moves from 0.773 to 0.766 and
the held-out calibrated figure from 0.827 to 0.825; `docs/band_decision.md`
is refreshed and lists both sets.

**Costs, stated.** The filter is two-sided and fitted on the full sample,
so the rolling and coverage evaluations carry a little look-ahead in this
one series, and the end-point factors will revise as data arrive. The
domestic share of UK CPI variance is now further from the paper than
before.

**Not done.** Seasonally adjusted euro-area and German CPI at source would
be the cleaner fix. It needs a new download, which this round did not make.

**Update, 19 Nov 2026 (Opus 5.5 (Fable limit)).** The ECB's working-day and
seasonally adjusted euro-area HICP was downloaded (`data/external/`). Over
1997Q2-2025Q2 the seasonal component it removes from the euro-area leg
correlates 0.95 with what this filter removes from the proxy, and a variant
built on it also passes the seasonal gate in every era
(`results/ecb_seasonal_check.md`). The production column stays as it is: the
two differ by at most 0.16pp in year-on-year world CPI inflation, a switch
would also need the pre-1997 German backcast and the China leg adjusted at
source, and Destatis's adjusted CPI was not obtainable without a GENESIS
login. Revisit at the next data refresh.

### 2. Estimation sample end: stays 2025Q1 until the next data refresh

**Decision.** Keep `--est-end 2025Q1` for the committed artefacts. Move it
to "last data quarter minus four" at the next data refresh (2026Q3 data),
in the same PR that regenerates everything on the new data.

**Why.** The paper reserves the last year of data. With data to 2026Q2
that rule gives 2025Q2; the repo's 2025Q1 was set when the data ended in
2026Q1 and was not moved at the 6 November refresh. Moving it now would
regenerate every artefact a second time on top of decision 1 and make the
before/after of the seasonal adjustment unattributable.

**Evidence that it is immaterial.** The replication was rerun on 18 Nov
with `--est-end 2025Q2` and nothing else changed (10,000 draws, seed
20260812, adjusted data; not committed):

| Posterior mean, 1 year ahead | 1992Q1–2025Q1 (committed) | 1992Q1–2025Q2 |
|---|---|---|
| Accepted draws / ESS | 731 / 333.3 | 687 / 310.8 |
| UK GDP: global / domestic / unidentified | 37.6 / 39.0 / 23.4% | 37.9 / 38.6 / 23.5% |
| UK CPI: global / domestic / unidentified | 52.0 / 24.8 / 23.2% | 53.1 / 25.7 / 21.2% |
| Monetary-policy share of UK GDP / UK CPI | 11.2 / 7.2% | 9.6 / 8.0% |

The largest move in a group share is 2.0 points (UK CPI unidentified),
against 68% credible intervals 25 to 35 points wide. No reading changes.

Reproduce: `python scripts/run_replication.py --draws 10000 --accepted 10000
--est-end 2025Q2` (it overwrites `results/summary.md`; restore it
afterwards).

### 3. November 2026 MPR comparator: not transcribed

The repo holds no November 2026 MPR figures: `results/forecast_vs_boe_obr.md`
compares against the July 2026 MPR and the March 2026 EFO, transcribed by
hand at the time. Transcribing the November MPR means fetching it, which
this round did not do. The comparison document says which vintage it uses.

**Update, 19 Nov 2026 (Opus 5.5 (Fable limit)): done.** Table 3.B of the
November MPR is in `data/external/boe_mpr_2026_11_table_3b.csv`, and the
comparison at its Q4 quarters is a new dated section of
`results/forecast_vs_boe_obr.md`. The older July sections are kept as they
were.

### 4. Scoring the 2026Q3 forecasts: not done

`data/boe_var_data.csv` ends in 2026Q2, so the 2026Q3 outturn is not in the
repo. Scoring needs a data download. When it is made, append the score as
a new record; do not rewrite the stored forecasts.

**Update, 19 Nov 2026 (Opus 5.5 (Fable limit)): done.** The first outturns
(ONS 12, 17 and 18 Nov; BoE IADB) are in `data/external/uk_outturns_2026q3.csv`,
and `results/score_2026q3.md` scores the two archived site rounds that
forecast 2026Q3, without changing them. GDP 1.94% and CPI 3.11% fall inside
the 68% band of both rounds; unemployment, 4.64%, is just inside the 90% band
of the 17 Nov run, which no round archived. One quarter is not evidence of
skill. `data/boe_var_data.csv` still ends in 2026Q2; the 2026Q3 data refresh
is a separate step.

### 5. UK CPI band coverage is quoted in the README

The README's "Known failures" quoted band coverage for UK GDP only. It now
also gives UK CPI: the raw 90% band for year-on-year UK CPI inflation
covers 76% of outturns one quarter ahead and 56% at eight quarters; the
68% band covers 56% and 46% (`results/coverage_evaluation.json`,
`production_spec.coverage_yoy`, `cpisa`). Before the seasonal adjustment
the 90% figures were 78% and 54%.

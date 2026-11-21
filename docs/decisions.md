# Decisions

Dated modelling decisions, the evidence behind each, and what was left out.
Newest first. `docs/band_decision.md` is the longer note for the 90% band
(issue #12); `docs/improvements.md` tracks the toolbox-review items.

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

### 4. Scoring the 2026Q3 forecasts: not done

`data/boe_var_data.csv` ends in 2026Q2, so the 2026Q3 outturn is not in the
repo. Scoring needs a data download. When it is made, append the score as
a new record; do not rewrite the stored forecasts.

### 5. UK CPI band coverage is quoted in the README

The README's "Known failures" quoted band coverage for UK GDP only. It now
also gives UK CPI: the raw 90% band for year-on-year UK CPI inflation
covers 76% of outturns one quarter ahead and 56% at eight quarters; the
68% band covers 56% and 46% (`results/coverage_evaluation.json`,
`production_spec.coverage_yoy`, `cpisa`). Before the seasonal adjustment
the 90% figures were 78% and 54%.

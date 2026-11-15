# Decision note: why the 90% bands under-cover (#12)

Written 10 November 2026 on 49 origins; refreshed 15 November 2026 on main's 50 origins (2012Q1–2024Q2, data to 2026Q2). Nothing here changes the model or any published number.

## Verdict

1. **The shortfall is mostly forecast medians missing in three episodes. It is not tail shape.** 87% of the 90% band's misses fall in three windows of target quarters, and inside each window the misses are on one side: 2014Q3–2016Q4 (90% below the band), 2020Q1–2021Q2 (86% below) and 2021Q3–2023Q4 (94% above). Outside those windows the raw 90% band covers 0.938.
2. **The raw shortfall is real.** Raw coverage is 0.769 with a block-bootstrap 95% interval of 0.64–0.86, which excludes 0.90.
3. **The shortfall after rescaling is too small for this evaluation to measure.** The held-out Gaussian-k coverage of 0.828 has a 95% interval of 0.68–0.95, which includes 0.90. Per-origin coverage has a lag-1 autocorrelation of 0.87, so the 50 origins carry only a handful of independent observations.

**Recommendation:** keep the published specification (Covid dummies, Gaussian k) and label the 90% fan with the measured coverage and its interval. Do not adopt LP scaling or either SV model as the default.

## Evidence

All figures are for the published specification (six Covid dummies), 50 origins 2012Q1–2024Q2, 8 horizons, 8 variables: 3,200 scored cells and 739 misses of the 90% band. `scripts/band_decision_tables.py` computes every number in this section and in the table under Explanation C from a `--dump-record` file (see Reproduce). The record was made on macOS (Apple Accelerate); the run reproduces the committed `results/coverage_evaluation.json` exactly on that machine. A run on another machine moves individual cells by up to 0.08 (#26), so the third decimals below are not platform-stable; none of the conclusions depends on them.

### Where the misses are

| Target quarters | Cells | 90% coverage | Share of all misses | Above / below band |
|---|---|---|---|---|
| 2012Q2–2014Q2 | 352 | 0.935 | 3% | 7 / 16 |
| 2014Q3–2016Q4 (oil price fall, low inflation) | 640 | 0.645 | 31% | 22 / 205 |
| 2017Q1–2019Q4 | 768 | 0.965 | 4% | 6 / 21 |
| 2020Q1–2021Q2 (Covid) | 384 | 0.641 | 19% | 19 / 119 |
| 2021Q3–2023Q4 (inflation) | 640 | 0.566 | 38% | 262 / 16 |
| 2024Q1–2026Q2 | 416 | 0.889 | 6% | 23 / 23 |
| **All** | 3,200 | **0.769** | 100% | 339 / 400 |

- Dropping only the 2022 target quarters lifts coverage to 0.796. Dropping 2021Q3–2023Q4 gives 0.820. So 2022 is the largest single episode but not the whole gap.
- 90% coverage in the inflation window by variable: cpisa 0.24, cpi_energy 0.26, world_cpi 0.32, bank_rate 0.53. All 153 misses of world_cpi, cpisa and bank_rate there are above the band.
- In 2014Q3–2016Q4: cpisa 0.31, cpi_energy 0.39, oil_price 0.39, world_cpi 0.55.
- In the Covid window: uk_gdp 0.06, world_gdp 0.15.
- The three CPI series (world_cpi, cpisa, cpi_energy) account for 54% of all misses; they cover 0.668 against 0.830 for the other five variables.
- A miss at one origin is followed by a miss at the next origin (same variable and horizon) 71% of the time, against a 23% unconditional miss rate.

### Explanation A: tail shape or volatility

This predicts a 68% band near nominal with a short 90% band, misses on both sides, heavy-tailed standardised errors, and a fix from stochastic volatility.

- **Against.** The 68% band is also short (0.615).
- **Against.** *(49-origin figures, not recomputed.)* Outside the Covid and inflation windows, errors scaled by the band width have kurtosis of 2.3–3.7 for seven variables (Bank Rate, at the lower bound, is 19.3). After removing each variable's mean and scale there, the bands cover 0.910 at 90% and 0.697 at 68%, which is what a Gaussian shape gives.
- **Against.** The pooled PIT has too much mass in both tails (0.105 above the 95th percentile, 0.124 below the 5th), which looks like fat tails. It is the sum of two opposite one-sided episodes. *(49-origin figures from the SV branch, not recomputed.)* Common SV with a random walk cut the upper share to 0.034, but the lower share stayed at 0.108, and it stayed between 0.108 and 0.131 in every specification in `common_sv_evaluation.json` and `variable_sv_evaluation.json`. Bands 2.3 times wider did not reach those misses. Those two files are not on main: the SV code was backed out in 364d407 and they survive at commit 227e4c1.
- **For.** Covid GDP is a volatility event: *(49-origin figures)* errors are 6.8 (world) and 11.7 (UK) band-implied standard deviations wide. That is 19% of the misses, and the published model already treats those quarters with dummies.

### Explanation B: the median is wrong for long stretches

This predicts both bands short, one-sided and persistent misses, and concentration in the variables whose level drifted.

- **For.** Every row of the table above.
- **For.** In the CPI series the below-band miss rate rises with horizon, from 0.11 at h1 to 0.24 at h8; the above-band rate goes from 0.13 to 0.16.
- **Caveat.** The evaluation scores cumulated levels, so one large shock produces same-side misses at later horizons and origins. One-sidedness alone does not prove a wrong conditional mean. At h1, where nothing is cumulated, misses are balanced (29 above, 28 below) and coverage is 0.858.
- A model with correct volatility would still miss a level that moves one way for two years; what closes the gap is a better conditional mean or wider bands everywhere. This matches the earlier finding on #12 that cpisa and cpi_energy stay at 0.76–0.77 even under common SV.

### Explanation C: the evaluation is too small

| Statistic | Estimate | Block bootstrap, 12-origin blocks | 95% interval |
|---|---|---|---|
| Raw 90% coverage, all origins | 0.769 | s.e. 0.059 | 0.64–0.86 |
| Gaussian k, rolling origin, 31–38 scored origins per cell | 0.828 | s.e. 0.069 | 0.68–0.95 |
| Raw 68% coverage, all origins | 0.615 | s.e. 0.067 | 0.48–0.74 |

- Per-origin coverage has a lag-1 autocorrelation of 0.87.
- The standard error still rises with block length (0.045, 0.055, 0.059 at 4, 8, 12 origins for the raw band), so these intervals are a lower bound on the uncertainty. The binomial standard error quoted in the results files is too small.
- *(49-origin figures, not recomputed.)* The gaps between specifications (0.767, 0.804, 0.855 raw for dummies, LP scaling and common SV) are about one standard error. They are paired comparisons on the same origins, so their own error is smaller, but the committed files hold no per-origin hits for the LP and SV runs, so it cannot be computed from them.

### Where the explanations converge

B explains where and when the bands fail, and A does not survive the one-sidedness or the PIT evidence except for Covid GDP. C limits what can be claimed: the raw band is short, but "under-covers by about 8pp after calibration" is not established, and neither is any ranking of LP against the SV variants on coverage.

## Recommended label for the 90% fan

> Central 90% band. In a 2012–2024 backtest on the level of each series, bands built this way contained the outturn 83% of the time (95% interval about 68% to 95%). Most misses came in 2014–16, 2020–21 and 2021–23, when the central forecast was off in one direction for several quarters.

- The backtest scores levels in model units; the site fan shows year-on-year growth (#27). The label says what was measured. If #27 adds a year-on-year coverage block, the label should quote that instead.
- It replaces "under-covers by about 8pp", which the interval does not support.

## What happened to PRs #19–#23

All five were merged as one stack into main on 10 November (#19, merge 7e06eff). Before that merge the opt-in common and variable SV code (#22, #23) was backed out (364d407) because its no-leakage tests compared runs bit for bit and could fail with BLAS ordering. So main has:

| PR | What it is | On main | Adopted as default |
|---|---|---|---|
| #19 held-out band calibration | Evaluation | Yes (`calibration_holdout`) | Yes: it is the source of the 0.828 |
| #20 conformal recalibration | Evaluation, records a negative result | Yes (`conformal_calibration`) | No: 0.828 against 0.828 for Gaussian k on the production specification, with a 90% band 25% wider and a CRPS about 5% worse |
| #21 Lenza–Primiceri Covid scaling | Opt-in model | Yes (`covid_volatility.py`, `results/covid_volatility_evaluation.json`) | No |
| #22 common SV | Opt-in model | No, backed out in 364d407 | No |
| #23 variable-specific SV | Opt-in model | No, backed out in 364d407 | No |

- None of the "not adopted" verdicts is a code defect. Restoring the SV code is a separate question from this note; it needs the test fix and a deterministic rerun, and a fix for #26 does not by itself address the test failures (those compare two runs in one process).
- The original note recorded an unexplained difference: the published specification's raw 90% coverage was 0.7640 in `coverage_evaluation.json` and 0.7666 in three later files with the same seeds, draws and paths. That size of difference is what #26 shows between machines: the random paths go through an eigen factor whose signs depend on the LAPACK build.

## Issue #12

Close it on this decision once the label is on the site. Two statements in its thread should be corrected when it closes: "the outer tails are fatter than any rescaling of a Gaussian band can fix" (the misses are one-sided episodes) and the fixed 8pp shortfall. The follow-up worth opening is on the conditional mean of the CPI block, since volatility work will not move the lower-tail misses.

## Reproduce

```
PYTHONPATH=src python scripts/run_coverage_evaluation.py --output /tmp/cov.json --dump-record /tmp/record.npz
PYTHONPATH=src python scripts/band_decision_tables.py /tmp/record.npz
```

The first command takes a few seconds; the second takes about a second. The tables use `prod_stretch90`, `prod_stretch68` and `prod_pit` from the record. The calibrated row uses the rolling-origin Gaussian factor from `run_coverage_evaluation._scale_factor`. The bootstrap resamples blocks of consecutive origins (10,000 replications).

# Decision note: why the 90% bands under-cover (#12)

10 November 2026. Branch `vahid/band-decision`, cut from `vahid/variable-sv` (#23, head `6c51680`). Nothing here changes the model or any published number.

## Verdict

1. **The shortfall is mostly forecast medians missing in three episodes. It is not tail shape.** 87% of the 90% band's misses fall in three windows of target quarters, and inside each window the misses are on one side: 2014Q3–2016Q4 (89% below the band), 2020Q1–2021Q2 (86% below) and 2021Q3–2023Q4 (95% above). Outside those windows the raw 90% band covers 0.935.
2. **The raw shortfall is real.** Raw coverage is 0.764 with a block-bootstrap 95% interval of 0.63–0.87, which excludes 0.90.
3. **The shortfall after rescaling is too small for this evaluation to measure.** The held-out Gaussian-k coverage of 0.822 has a 95% interval of 0.67–0.95, which includes 0.90. The 49 origins carry about 7 independent observations.

**Recommendation:** keep the published specification (Covid dummies, Gaussian k) and label the 90% fan with the measured coverage and its interval. Do not adopt LP scaling or either SV model as the default. Land #19–#23 as evidence and opt-in code.

## Evidence

All figures are for the published specification (six Covid dummies), 49 origins 2012Q1–2024Q1, 8 horizons, 8 variables: 3,136 scored cells and 740 misses of the 90% band. I re-ran `scripts/run_coverage_evaluation.py --dump-record` on this branch (3 seconds). The output is identical to the committed `results/coverage_evaluation.json`, and the per-origin arrays are what the tables below use.

### Where the misses are

| Target quarters | Cells | 90% coverage | Share of all misses | Above / below band |
|---|---|---|---|---|
| 2012Q2–2014Q2 | 352 | 0.935 | 3% | 6 / 17 |
| 2014Q3–2016Q4 (oil price fall, low inflation) | 640 | 0.642 | 31% | 25 / 204 |
| 2017Q1–2019Q4 | 768 | 0.965 | 4% | 6 / 21 |
| 2020Q1–2021Q2 (Covid) | 384 | 0.643 | 19% | 19 / 118 |
| 2021Q3–2023Q4 (inflation) | 640 | 0.564 | 38% | 264 / 15 |
| 2024Q1–2026Q1 | 352 | 0.872 | 6% | 23 / 22 |
| **All** | 3,136 | **0.764** | 100% | 343 / 397 |

- Dropping only the 2022 target quarters lifts coverage to 0.791. Dropping 2021Q3–2023Q4 gives 0.815. So 2022 is the largest single episode but not the whole gap.
- 90% coverage in the inflation window by variable: cpisa 0.24, cpi_energy 0.26, world_cpi 0.33, bank_rate 0.51. All 154 misses of world_cpi, cpisa and bank_rate there are above the band.
- In 2014Q3–2016Q4: cpisa 0.35, cpi_energy 0.38, oil_price 0.39, world_cpi 0.54.
- In the Covid window: uk_gdp 0.06, world_gdp 0.15.
- The three CPI series (world_cpi, cpisa, cpi_energy) account for 53% of all misses; they cover 0.663 against 0.824 for the other five variables.
- A miss at one origin is followed by a miss at the next origin (same variable and horizon) 71% of the time, against a 24% unconditional miss rate.

### Explanation A: tail shape or volatility

This predicts a 68% band near nominal with a short 90% band, misses on both sides, heavy-tailed standardised errors, and a fix from stochastic volatility.

- **Against.** The 68% band is also short (0.611).
- **Against.** Outside the Covid and inflation windows, errors scaled by the band width have kurtosis of 2.3–3.7 for seven variables (Bank Rate, at the lower bound, is 19.3). After removing each variable's mean and scale there, the bands cover 0.910 at 90% and 0.697 at 68%, which is what a Gaussian shape gives.
- **Against.** The pooled PIT has too much mass in both tails (0.107 above the 95th percentile, 0.125 below the 5th), which looks like fat tails. It is the sum of two opposite one-sided episodes. Common SV with a random walk cuts the upper share to 0.034, but the lower share stays at 0.108, and it stays between 0.108 and 0.131 in every specification in `common_sv_evaluation.json` and `variable_sv_evaluation.json`. Bands 2.3 times wider do not reach those misses.
- **For.** Covid GDP is a volatility event: errors are 6.8 (world) and 11.7 (UK) band-implied standard deviations wide. That is 19% of the misses, and the published model already treats those quarters with dummies.

### Explanation B: the median is wrong for long stretches

This predicts both bands short, one-sided and persistent misses, and concentration in the variables whose level drifted.

- **For.** Every row of the table above.
- **For.** In the CPI series the below-band miss rate rises with horizon, from 0.10 at h1 to 0.24 at h8; the above-band rate goes from 0.13 to 0.16.
- **Caveat.** The evaluation scores cumulated levels, so one large shock produces same-side misses at later horizons and origins. One-sidedness alone does not prove a wrong conditional mean. At h1, where nothing is cumulated, misses are balanced (29 above, 27 below) and coverage is 0.857.
- A model with correct volatility would still miss a level that moves one way for two years; what closes the gap is a better conditional mean or wider bands everywhere. This matches the earlier finding on #12 that cpisa and cpi_energy stay at 0.76–0.77 even under common SV.

### Explanation C: the evaluation is too small

| Statistic | Estimate | Block bootstrap, 12-origin blocks | 95% interval |
|---|---|---|---|
| Raw 90% coverage, all origins | 0.764 | s.e. 0.061 | 0.63–0.87 |
| Gaussian k, rolling origin, 37 scored origins | 0.822 | s.e. 0.072 | 0.67–0.95 |
| Raw 68% coverage, all origins | 0.611 | s.e. 0.071 | 0.47–0.75 |

- Per-origin coverage has a lag-1 autocorrelation of 0.87, which leaves about 7 effective origins out of 49.
- The standard error still rises with block length (0.045, 0.057, 0.061 at 4, 8, 12 origins for the raw band), so these intervals are a lower bound on the uncertainty. The binomial 0.043 quoted in the results files is too small.
- The gaps between specifications (0.767, 0.804, 0.855 raw) are about one standard error. They are paired comparisons on the same origins, so their own error is smaller, but the committed files hold no per-origin hits for the LP and SV runs and I could not compute it.

### Where the explanations converge

B explains where and when the bands fail, and A does not survive the one-sidedness or the PIT evidence except for Covid GDP. C limits what can be claimed: the raw band is short, but "under-covers by about 8pp after calibration" is not established, and neither is any ranking of LP against the SV variants on coverage.

## Recommended label for the 90% fan

> Central 90% band. In a 2012–2024 backtest on the level of each series, bands built this way contained the outturn 82% of the time (95% interval about 67% to 95%). Most misses came in 2014–16, 2020–21 and 2021–23, when the central forecast was off in one direction for several quarters.

- The backtest scores levels in model units; the site fan shows year-on-year growth. The label says what was measured.
- It replaces "under-covers by about 8pp", which the interval does not support.

## PRs #19–#23

The stack is #19 ← #20 ← #21 ← #22 ← #23, with #19 based on #18. Each head has green lint and fast tests; the slow validation job was skipped on all five.

| PR | What it is | Correctness bug | Adopted | Action |
|---|---|---|---|---|
| #19 held-out band calibration | Evaluation only | None found | Yes: it is the source of the 0.822 | Land |
| #20 conformal recalibration | Evaluation, records a negative result | None found | Not recommended (0.824 against 0.822, wider, worse CRPS) | Land as evidence |
| #21 Lenza–Primiceri Covid scaling | Opt-in model | None found | Not recommended as default (0.803 raw, CRPS 4% worse) | Land as opt-in |
| #22 common SV | Opt-in model | Test only: `test_common_sv.py` no-leakage tests compare bit for bit and can fail with BLAS ordering; the tolerance fix `6c51680` is only on #23 | Not recommended (0.855 raw with 2.3× width, CRPS 9% worse) | Land together with #23, not alone |
| #23 variable-specific SV | Opt-in model | None found | Not recommended (CRPS 1.030–1.076, width 1.9–3.9×) | Land; carries the #22 test fix |

- "None found" means: CI is green on the head, and the coverage rerun reproduces the committed JSON exactly. I did not re-review the LP or SV sampler code this session.
- No PR should be closed. None of the "not recommended" verdicts is a code defect.
- One unexplained difference: the published specification's raw 90% coverage is 0.7640 in `coverage_evaluation.json` and 0.7666 in the three later files, which say they use the same seeds, draws and paths. It is 8 cells out of 3,136 and does not affect any conclusion. This note uses 0.764.

## Issue #12

Close it on this decision once the label is on the site. Two statements in its thread should be corrected when it closes: "the outer tails are fatter than any rescaling of a Gaussian band can fix" (the misses are one-sided episodes) and the fixed 8pp shortfall. The follow-up worth opening is on the conditional mean of the CPI block, since volatility work will not move the lower-tail misses.

## Reproduce

```
PYTHONPATH=src python scripts/run_coverage_evaluation.py --output /tmp/cov.json --dump-record /tmp/record.npz
```

The tables use `prod_stretch90`, `prod_stretch68` and `prod_pit` from that file: a cell is inside the band when its stretch is at most 1, and the PIT gives the side of a miss. The calibrated row uses `gaussian_rolling_factors` from the same script. The bootstrap resamples blocks of consecutive origins (10,000–20,000 replications).

# Replication spec — BoE Macro Technical Paper No. 3
"A structural VAR model for the UK economy", Brignone & Piffer (July 2025).
PDF: `docs/a-structural-var-model-for-the-uk-economy.pdf`

## Goal
Python replication of the paper's Bayesian SVAR: estimation, identification,
IRFs, FEVDs, estimated shocks, historical decompositions. Compare qualitatively
to Figures 2–6 of the paper.

## Model
- 8 quarterly variables (Table 1), sample for estimation: **1992Q1–2023Q2**
  (data through 2024Q2 kept for later exercises).
- VAR(p) in (mostly) 100·log levels with constant + 6 Covid dummies
  (one per quarter 2020Q1–2021Q2). Use p = 4 lags (paper doesn't state; 4 is
  standard for quarterly; make it a parameter).
- Variables and transforms (order fixed — used by zero restrictions):
  1. Real World GDP (100·log) — UK-trade-weighted; proxy: OECD/IMF weighted agg.
  2. World CPI (100·log) — UK-trade-weighted; proxy likewise.
  3. Real oil price in sterling (100·log) — Brent USD → GBP, deflated by UK CPI.
  4. Bank Rate (level, %).
  5. Sterling ERI (100·log) — BoE effective exchange-rate index.
  6. UK CPISA (100·log) — seasonally adjusted CPI.
  7. UK CPI Energy (100·log).
  8. UK Real GDP (100·log).
- Estimation: Bayesian, Minnesota (Giannone-Lenza-Primiceri 2015 style NIW)
  prior + sum-of-coefficients prior; Covid dummies as exogenous regressors
  (pandemic-prior spirit of Cascaldi-Garcia 2022 is acceptable to simplify).
- Identification: B = chol(Σ)·Q, Q drawn per Arias–Rubio-Ramírez–Waggoner
  (2018) so zero restrictions hold exactly; sign restrictions accept/reject.
  Target: 1,000+ accepted draws (paper uses 10,000).

## Identifying restrictions (Table 2, impact only)
Rows = variables (order above), columns = shocks:
world demand (WD), world energy (WE), world supply (WS), unidentified-global
(U1), UK demand (UD), UK supply (US), UK mon. pol. (MP), unidentified-UK (U2).

| Variable    | WD | WE | WS | U1 | UD | US | MP | U2 |
|-------------|----|----|----|----|----|----|----|----|
| World GDP   | +  | +  | +  |    | 0  | 0  | 0  | 0  |
| World CPI   | +  | −  | −  |    | 0  | 0  | 0  | 0  |
| Oil price   |    | −  |    |    | 0  | 0  | 0  | 0  |
| Bank Rate   |    |    |    |    | +  |    | +  |    |
| Exch. rate  |    |    |    |    |    |    |    |    |
| UK CPISA    | +  | −  | −  |    | +  | −  | −  |    |
| CPI Energy  | +  | −  | 0  |    |    |    |    |    |
| Real UK GDP | +  | +  | +  |    | +  | +  | −  |    |

(U1 has no sign restrictions but is a "global" shock — zero block on it is NOT
imposed on world rows; U2 obeys the same zero block as UK shocks. Blank = unrestricted.)

## Module API (code to this — src/boe_var/)
- `data.py`: `load_data() -> pd.DataFrame` — index PeriodIndex('Q'), columns
  `["world_gdp","world_cpi","oil_price","bank_rate","eri","cpisa","cpi_energy","uk_gdp"]`,
  already transformed (100·log except bank_rate). Reads `data/boe_var_data.csv`.
- `bvar.py`: `class BVAR(y: np.ndarray, lags: int, dummies: np.ndarray | None)`
  with `.sample_posterior(n_draws) -> list[PosteriorDraw]` where PosteriorDraw
  has `.Pi` (k*(k·p+1+n_dummy) coeffs), `.Sigma` (k×k), and helper
  `.companion()`.
- `identification.py`: `draw_Q(Sigma, restrictions) -> np.ndarray | None`
  (None if rejected) and `identify(draws, restrictions, target_accepted)` →
  list of (draw, B). `restrictions` defined as module-level constants
  `ZERO_RESTRICTIONS`, `SIGN_RESTRICTIONS` (dict/array encoding table above).
- `analysis.py`: `irf(draw, B, horizons)`, `fevd(...)`, `historical_decomposition(...)`,
  `estimated_shocks(...)`; plotting helpers writing PNGs to `results/`.
- `scripts/run_replication.py`: end-to-end — load data, estimate, identify,
  produce `results/fig2_irf_world.png`, `fig3_irf_uk.png`, `fig4_fevd.png`,
  `fig5_shocks.png`, `fig6_hist_decomp.png` + `results/summary.md`.

## Environment
Conda env `python313`. Deps: numpy, scipy, pandas, matplotlib, requests.
Add `pyproject.toml` (package `boe_var`, src layout) and `requirements.txt`.

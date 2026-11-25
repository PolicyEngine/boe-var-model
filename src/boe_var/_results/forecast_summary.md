# Forecast-revision exercise (Section 5)

- Estimation sample 1992Q1–2025Q1; T = 2026Q2, T−1 = 2026Q1; lags 4, horizons 13.
- Posterior draws 6000, accepted 430 (7.2%), importance-weight ESS 185.6.

## P(sign) of the identified shocks at T = 2026Q2 (weighted)

| Shock | P(>0) | P(<0) |
|---|---|---|
| World demand | 0.93 | 0.07 |
| World energy | 0.03 | 0.97 |
| World supply | 0.55 | 0.45 |
| UK demand | 0.10 | 0.90 |
| UK supply | 0.72 | 0.28 |
| UK mon. pol. | 0.68 | 0.32 |

## Composite impulse response (median, YoY)

- YoY CPI inflation: peak effect +0.54 pp at h = 3 quarters after 2026Q2.
- YoY GDP growth: peak effect -0.34 pp at h = 6 quarters after 2026Q2.
- Basis: year-on-year rates here and in figures 1, 8 and 9 are four-quarter log differences (100 × the change in the log level). The simple rate that ONS, the BoE and the OBR publish is higher by about x²/200 (0.04 pp at 2.8%, 0.5 pp at 10%); convert with `forecast.log_to_simple_pct` before comparing.

## Adding-up check (Figure 9 identity)

- forecast_T = pseudo_{T−1} + composite IRF, exact per draw: max abs error across all draws/horizons/variables = 8.19e-12.

Figures: fig1_forecast.png, fig7_shock_distributions.png, fig8_composite_irf.png, fig9_forecast_revision.png.

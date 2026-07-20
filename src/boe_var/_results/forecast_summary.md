# Forecast-revision exercise (Section 5)

- Estimation sample 1992Q1–2025Q1; T = 2026Q1, T−1 = 2025Q4; lags 4, horizons 13.
- Posterior draws 6000, accepted 430 (7.2%), importance-weight ESS 197.9.

## P(sign) of the identified shocks at T = 2026Q1 (weighted)

| Shock | P(>0) | P(<0) |
|---|---|---|
| World demand | 0.29 | 0.71 |
| World energy | 0.10 | 0.90 |
| World supply | 0.87 | 0.13 |
| UK demand | 0.81 | 0.19 |
| UK supply | 0.71 | 0.29 |
| UK mon. pol. | 0.16 | 0.84 |

## Composite impulse response (median, YoY)

- YoY CPI inflation: peak effect +0.22 pp at h = 3 quarters after 2026Q1.
- YoY GDP growth: peak effect +0.27 pp at h = 0 quarters after 2026Q1.

## Adding-up check (Figure 9 identity)

- forecast_T = pseudo_{T−1} + composite IRF, exact per draw: max abs error across all draws/horizons/variables = 5.23e-12.

Figures: fig1_forecast.png, fig7_shock_distributions.png, fig8_composite_irf.png, fig9_forecast_revision.png.

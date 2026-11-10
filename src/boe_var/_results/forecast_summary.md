# Forecast-revision exercise (Section 5)

- Estimation sample 1992Q1–2025Q1; T = 2026Q2, T−1 = 2026Q1; lags 4, horizons 13.
- Posterior draws 6000, accepted 455 (7.6%), importance-weight ESS 181.4.

## P(sign) of the identified shocks at T = 2026Q2 (weighted)

| Shock | P(>0) | P(<0) |
|---|---|---|
| World demand | 0.95 | 0.05 |
| World energy | 0.02 | 0.98 |
| World supply | 0.28 | 0.72 |
| UK demand | 0.05 | 0.95 |
| UK supply | 0.76 | 0.24 |
| UK mon. pol. | 0.79 | 0.21 |

## Composite impulse response (median, YoY)

- YoY CPI inflation: peak effect +0.66 pp at h = 3 quarters after 2026Q2.
- YoY GDP growth: peak effect -0.36 pp at h = 6 quarters after 2026Q2.

## Adding-up check (Figure 9 identity)

- forecast_T = pseudo_{T−1} + composite IRF, exact per draw: max abs error across all draws/horizons/variables = 3.64e-12.

Figures: fig1_forecast.png, fig7_shock_distributions.png, fig8_composite_irf.png, fig9_forecast_revision.png.

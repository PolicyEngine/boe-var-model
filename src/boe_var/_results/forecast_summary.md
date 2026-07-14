# Forecast-revision exercise (Section 5)

- Estimation sample 1992Q1–2023Q2; T = 2024Q2, T−1 = 2024Q1; lags 4, horizons 13.
- Posterior draws 6000, accepted 487 (8.1%), importance-weight ESS 207.5.

## P(sign) of the identified shocks at T = 2024Q2 (weighted)

| Shock | P(>0) | P(<0) |
|---|---|---|
| World demand | 0.81 | 0.19 |
| World energy | 0.20 | 0.80 |
| World supply | 0.34 | 0.66 |
| UK demand | 0.39 | 0.61 |
| UK supply | 0.75 | 0.25 |
| UK mon. pol. | 0.56 | 0.44 |

## Composite impulse response (median, YoY)

- YoY CPI inflation: peak effect -0.22 pp at h = 0 quarters after 2024Q2.
- YoY GDP growth: peak effect +0.40 pp at h = 1 quarters after 2024Q2.

## Adding-up check (Figure 9 identity)

- forecast_T = pseudo_{T−1} + composite IRF, exact per draw: max abs error across all draws/horizons/variables = 5.29e-12.

Figures: fig1_forecast.png, fig7_shock_distributions.png, fig8_composite_irf.png, fig9_forecast_revision.png.

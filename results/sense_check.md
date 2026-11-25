# Sense-check against the BoE paper

17 November 2026, at the committed run (`results/summary.md`: 10,000
posterior draws, 731 accepted, ESS 333.3; data to 2026Q2, estimation
1992Q1–2025Q1; world CPI seasonally adjusted, see `data/README.md`).
Replaces the 13 July 2026 version, which described a 186-draw run.

Comparison is qualitative: the paper publishes figures, not tables, so
paper values below are approximate read-offs from its Figures 2–6
(PDF pp. 10–18).

## What matches

| Check | Result |
|---|---|
| Identification (Table 2) | `identification.py` matches every zero and sign; exchange rate unrestricted |
| Fig 2–3 signs | All restricted signs hold; oil price rises after a world supply shock (the paper's unrestricted check) |
| Fig 2–3 magnitudes | Same order as the paper: monetary-policy shock moves Bank Rate about 10bp on impact (paper about 8–10bp); UK supply raises GDP about 0.2 (paper 0.2–0.3); world energy moves UK GDP about 0.45 at h20 (paper about 0.5). Medians of the weighted IRFs, checked 17 Nov |
| Fig 5 episodes | 2008–09 world demand collapse, 2022 contractionary energy shocks, monetary loosening in 2022 then tightening in 2023, muted 2020–21 shocks (Covid dummies) |
| Fig 6 | Dated axis; Covid-dummy contribution shown as its own (black) component, as in the paper |
| UK GDP global share, 1 year | 37.6% (posterior mean; 68% band 23–50%) against the paper's about 40% |

## What does not match (open)

| Item | Here | Paper | Note |
|---|---|---|---|
| UK CPI global share, 1 year | 52.0% (68% band 34–68%) | about 50% | Close on the mean after the world CPI seasonal adjustment (was 41.5%); the band is wide |
| UK CPI domestic share, 1 year | 24.8% | about 40% | Further from the paper than before the adjustment (36.8%) |
| Monetary policy share of UK GDP, 1 year | 11.2% | about 20% (largest domestic) | The clearest open discrepancy; here UK supply (14.9%) and UK demand (12.9%) are larger |
| Monetary-policy shock, GDP response | −0.14 on impact, persisting: about −0.16 at h8 and −0.17 at h20 | −0.12 on impact, fading to −0.07 by h20 | Close on impact; the paper's response fades and the level effect here does not (unit root, `README.md` Known failures). Medians from `results/irf_vs_paper.md` |
| Monetary-policy shock, CPI at h20 | about −0.21 | about −0.4 | Smaller here |
| World supply → UK GDP median | +0.12 to +0.14 early, crosses zero around h12 | stays positive (+0.1 to +0.2) | Not a low-draw artefact any more (731 accepted draws); bands cover the paper's path |

Candidate explanations, none tested: the proxy world GDP and CPI series
(the Bank's UK-trade-weighted series are unpublished), the 2026Q2 data
vintage against the paper's August 2024 one, and the prior settings
(`docs/methodology.md` section 7).

## Bands

The paper uses 10,000 accepted draws; this run has 731 (ESS 333), so the
plotted bands carry more Monte Carlo noise than the paper's.

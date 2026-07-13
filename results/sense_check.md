# Final sense-check vs the BoE paper (post-identification-fix figures)

Independent panel-by-panel comparison of `results/fig2..fig6` and
`summary.md` against Figures 2–6 and the quantitative claims of the paper
(pp. 10–18).

## Verdicts

| Figure | Verdict |
|---|---|
| Fig 2 (world IRFs) | PASS with caveats |
| Fig 3 (UK IRFs) | PASS |
| Fig 4 (FEVD) | PASS with caveats |
| Fig 5 (estimated shocks) | PASS |
| Fig 6 (historical decomposition) | CAVEAT (presentation) |
| summary.md internal consistency | PASS (rows sum to ~100; headline = detailed sums) |

Matches: all Table 2 sign/zero patterns visible in the IRFs; world-energy row
of Fig 2 clean across all 8 panels; oil price rises after a world supply shock
(the paper's key unrestricted check); explained variance ~80% domestic / ~90%
global; global share of UK variance at 1y 42.5% GDP / 52.2% CPI vs paper
~40%/~50%; Fig 5 reproduces the paper's narrative episodes (2008 world-demand
collapse, 2022 contractionary energy, post-2022 UK monetary loosening then
2023 tightening, muted shocks 2020–21 from Covid dummies).

## Caveats / open items

1. Fig 2: UK GDP medians to world demand and world supply shocks decay
   through zero at long horizons (paper: persistently positive; bands cover
   the paper's path). Likely low-draw / proxy-world-data artifact.
2. Fig 4: within the world supply side, energy > supply for World GDP and oil
   price here, while the paper's figure suggests supply is the larger
   long-horizon contributor. The paper's claims are about the combined
   supply side, which we match.
3. Known, disclosed: UK monetary policy is not the largest domestic
   contributor to CPI variance (UK supply 15.6% > mon. pol. 7.4%).
4. Fig 6 presentation: bare quarter-index x-axis (should be dates), and Covid
   dummies folded into the deterministic component instead of shown as the
   paper's separate black bars — understates the structural-shock share of
   the 2021–23 inflation surge visually. See docs/improvements.md item 8.
5. Bands are wider than the paper's (186 accepted draws, ESS 91, vs the
   paper's 10,000 accepted). Raise --draws for production-quality figures.

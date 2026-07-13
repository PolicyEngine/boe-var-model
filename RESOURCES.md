# Resources

Annotated links used for this replication. All URLs checked 2026-07-13;
notes flag any that block automated fetching (they resolve fine in a
browser).

## The paper

- **Landing page (Bank of England):**
  https://www.bankofengland.co.uk/macro-technical-paper/2025/a-structural-var-model-for-the-uk-economy
  — Macro Technical Paper No. 3, Brignone & Piffer, July 2025. (BoE site
  returns 403 to non-browser clients; URL is canonical.)
- **PDF:** local copy at `docs/a-structural-var-model-for-the-uk-economy.pdf`;
  the Bank serves it from the landing page above.
- **SSRN version:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5377807
  (SSRN also blocks scrapers; resolves in a browser.)

## Companion work by the authors

- **Brignone & Piffer, structural forecast analysis companion paper:**
  https://michelepiffer.github.io/pdf_research/BrignonePiffer_StucturalForecastAnalysis.pdf
  — develops the structural decomposition of forecast revisions used in
  Section 5 of the MTP. (Note the "Stuctural" typo is in the actual
  filename; verified live.)
- **Michele Piffer — codes:** https://michelepiffer.github.io/#codes
  — MATLAB code for Bayesian SVAR identification (sign restrictions,
  importance sampling), useful as a cross-check for the identification step.

## Reference implementations

- **BEAR toolbox (ECB):** https://github.com/european-central-bank/BEAR-toolbox
  — MATLAB Bayesian VAR suite (Minnesota/NIW priors, sign/zero
  identification, IRFs, FEVDs, historical decompositions). Good reference
  for prior hyperparameters and posterior algebra.
- **VAR-Toolbox (Ambrogio Cesa-Bianchi):** https://github.com/ambropo/VAR-Toolbox
  — MATLAB VAR/local-projection toolbox with seven identification schemes
  incl. combined zero/sign restrictions; clear, readable implementations.
- **bsvarSIGNs (R):** https://github.com/bsvars/bsvarSIGNs — Bayesian SVARs
  with sign, zero and narrative restrictions; implements exactly the
  Arias et al. (2018) machinery this replication needs. Docs:
  https://bsvars.org/bsvarSIGNs/
- **Arias et al. (2018) Econometrica page + supplemental material:**
  https://www.econometricsociety.org/publications/econometrica/2018/03/01/inference-based-structural-vector-autoregressions-identified
  — the supplement includes step-by-step pseudo-code for Algorithms 1 and 3.
  (Econometric Society site 403s to scrapers; URL confirmed via the
  Society's own search index. Wiley DOI: https://doi.org/10.3982/ECTA14468)
- **Jonas Arias — personal site:** https://sites.google.com/site/jonasarias/home
  — links to replication code for the zero/sign restriction algorithms.
- **Haroon Mumtaz — example code:** https://sites.google.com/site/hmumtaz77/code
  — MATLAB/Julia teaching code for Bayesian VARs, Gibbs sampling, sign
  restrictions; useful pedagogical cross-checks.

## Method papers

- **Giannone, Lenza & Primiceri (2015), "Prior Selection for Vector
  Autoregressions", REStat:** https://doi.org/10.1162/REST_a_00483 —
  hierarchical Minnesota/NIW prior with sum-of-coefficients; the paper's
  stated prior framework.
- **Arias, Rubio-Ramírez & Waggoner (2018), "Inference Based on Structural
  Vector Autoregressions Identified With Sign and Zero Restrictions",
  Econometrica:** https://doi.org/10.3982/ECTA14468 — the Q-draw algorithm
  used for identification.
- **Uhlig (2005), "What are the effects of monetary policy on output?",
  JME:** https://doi.org/10.1016/j.jmoneco.2004.05.007 — the classic
  pure-sign-restriction agnostic identification approach.
- **Cascaldi-Garcia (2022), "Pandemic Priors", Fed IFDP 1352:**
  https://www.federalreserve.gov/econres/ifdp/pandemic-priors.htm
  (DOI: https://doi.org/10.17016/IFDP.2022.1352) — Covid dummy-observation
  extension of the Minnesota prior; the paper follows this approach.
- **Chan, Matthes & Yu (2025), "Large Structural VARs with Multiple Sign
  and Ranking Restrictions":** https://arxiv.org/abs/2503.20668 — the
  column-reordering/sign-flipping trick the authors combine with Arias et
  al. (2018) to speed up accept/reject sampling of Q.

## Data sources

- **ONS time series API:** https://www.ons.gov.uk/timeseries — UK CPI
  (D7BT / CPISA), CPI energy components, real GDP (ABMI). Series are
  fetchable as JSON/CSV via
  `https://api.ons.gov.uk/timeseries/{id}/dataset/{ds}/data`.
- **Bank of England IADB (Interactive Database):**
  https://www.bankofengland.co.uk/boeapps/database — Bank Rate and the
  sterling effective exchange-rate index (ERI, series XUQABK67 and
  relatives); CSV download supported.
- **FRED (St. Louis Fed):** https://fred.stlouisfed.org — Brent oil price
  (series DCOILBRENTEU), USD/GBP exchange rate, and OECD/world aggregates
  used to proxy the Bank's internal trade-weighted world GDP/CPI series.

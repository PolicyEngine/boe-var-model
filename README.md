# boe-var-model

Python replication of **Bank of England Macro Technical Paper No. 3** —
*"A structural VAR model for the UK economy"*, Davide Brignone and Michele
Piffer (July 2025).

The paper documents the Bayesian structural VAR the Bank uses to disentangle
domestic and global drivers of the UK business cycle: eight quarterly
variables (three global, five UK), identified with zero and sign restrictions
into six structural shocks (world demand, world energy, world supply, UK
demand, UK supply, UK monetary policy) plus two unidentified residual shocks.
This repo re-implements the pipeline in Python: data assembly, Bayesian
estimation with a Minnesota/NIW prior, Arias–Rubio-Ramírez–Waggoner (2018)
identification, and the standard outputs (IRFs, FEVDs, estimated shocks,
historical decompositions), compared qualitatively to Figures 2–6 of the
paper.

See `SPEC.md` for the full replication spec and `docs/methodology.md` for a
writeup of the method and of where this replication deviates from the paper.

## Repository layout

```
├── SPEC.md                 # replication spec (model, restrictions, API)
├── docs/
│   ├── a-structural-var-model-for-the-uk-economy.pdf   # the paper
│   ├── methodology.md      # method writeup + differences from the paper
│   └── improvements.md     # ranked list of fidelity gaps
├── src/boe_var/            # package (src layout)
│   ├── data.py             # load_data() -> transformed quarterly DataFrame
│   ├── bvar.py             # BVAR class: NIW posterior sampling
│   ├── identification.py   # zero/sign restrictions, draw_Q, identify
│   ├── analysis.py         # IRFs, FEVDs, shocks, historical decompositions
│   ├── forecast.py         # forecasts, fan bands, forecast revisions
│   ├── evaluation.py       # rolling-origin point and band evaluation
│   └── unemployment_satellite.py  # GDP growth -> unemployment mapping
├── scripts/
│   ├── download_data.py    # fetch raw series -> data/boe_var_data.csv
│   ├── run_replication.py  # estimate, identify, write fig2-fig6 + summary.md
│   ├── run_forecast_revision.py   # paper section 5: fig1, fig7-fig9, forecast_summary.md
│   ├── run_rolling_evaluation.py  # results/rolling_evaluation.json
│   ├── run_coverage_evaluation.py # results/coverage_evaluation.json
│   ├── run_satellite_validation.py
│   └── run_external_comparison.py # numbers in results/forecast_vs_boe_obr.md
├── data/                   # raw + assembled data (raw data git-ignored)
├── results/                # figures, summaries and evaluation JSON
├── tests/
├── RESOURCES.md            # annotated links: paper, code, methods, data
└── pyproject.toml          # package boe_var
```

## Quickstart

Python 3.11 or later. With [uv](https://docs.astral.sh/uv/) (what CI uses):

```bash
uv sync
uv run python scripts/download_data.py     # builds data/boe_var_data.csv
uv run python scripts/run_replication.py   # writes figures + summary to results/
uv run pytest -q -m "not calibration and not slow"
```

`pip install -e .` in any Python 3.11+ environment also works.

Outputs land in `results/`: `fig2_irf_world.png`, `fig3_irf_uk.png`,
`fig4_fevd.png`, `fig5_shocks.png`, `fig6_hist_decomp.png`, `summary.md`.

## Toolboxes that can replicate the paper's method (zero + sign restrictions, Minnesota prior)

| Resource | Exact URL |
|---|---|
| BEAR toolbox (ECB, MATLAB — Bayesian VAR with sign/zero restrictions, used widely in central banks) | https://github.com/european-central-bank/BEAR-toolbox |
| BEAR on MATLAB File Exchange | https://www.mathworks.com/matlabcentral/fileexchange/103370-bear |
| Cesa-Bianchi VAR Toolbox (MATLAB — IRFs, FEVD, historical decompositions, sign restrictions; he's a BoE economist cited in the paper) | https://github.com/ambropo/VAR-Toolbox |
| bsvarSIGNs (R — directly implements Arias–Rubio-Ramírez–Waggoner 2018, the exact algorithm this paper uses) | https://github.com/bsvars/bsvarSIGNs (docs: https://bsvars.org/bsvarSIGNs/) |
| Arias et al. (2018) original replication files | Econometrica page (zip in "Supplemental Material"): https://www.econometricsociety.org/publications/econometrica/2018/03/01/inference-based-structural-vector-autoregressions-identified ; Jonas Arias' site: https://sites.google.com/site/jonasarias/home |
| Haroon Mumtaz — Bayesian econometrics for central bankers, incl. sign-restriction VAR MATLAB code | https://sites.google.com/site/hmumtaz77/code |
| Michele Piffer's own code page (paper co-author) | https://michelepiffer.github.io/#codes |

## The paper itself (no official replication package released)

- BoE page: https://www.bankofengland.co.uk/macro-technical-paper/2025/a-structural-var-model-for-the-uk-economy
- BoE paper PDF: https://www.bankofengland.co.uk/-/media/boe/files/macro-technical-paper/2025/a-structural-var-model-for-the-uk-economy.pdf
- Companion paper PDF: https://michelepiffer.github.io/pdf_research/BrignonePiffer_StucturalForecastAnalysis.pdf

See `RESOURCES.md` for the full annotated list (method papers, data sources).

## Caveats

- **World aggregates are proxies.** The paper uses the Bank's internal
  UK-trade-weighted world GDP and world CPI series, which are not published.
  We proxy them with weighted aggregates from OECD/IMF sources, so global
  blocks will not match the paper exactly.
- **No official replication package exists.** The Bank has not released code
  or data for MTP No. 3; this replication is reconstructed from the paper's
  description and standard references, and can only be validated
  qualitatively against the published figures.
- The paper does not state the lag length; we assume p = 4 (a parameter).
- Fewer accepted identification draws: the committed figures use 744
  accepted draws out of 10,000 posterior draws (importance-weight ESS 356,
  `results/summary.md`), and the script default is 200 (`--accepted`),
  against the paper's 10,000 accepted. The pandemic-prior treatment is
  simplified (Covid dummies as exogenous regressors). See
  `docs/methodology.md` for the full list of differences.

## Known failures

Read these before quoting a forecast, a band or an impulse response.

- **No forecast skill over simple benchmarks.** In the rolling-origin
  evaluation (50 origins, 2026Q2 data), no variable × horizon is significantly more
  accurate than a random walk, a random walk with drift or an AR(1) once the 64 tests per benchmark are
  adjusted together (Benjamini–Hochberg q < 0.05 in 0 of 64;
  `results/rolling_evaluation.json`, `multiplicity`).
- **The 90% bands are too narrow.** Raw production-spec 90% bands cover
  88% of UK GDP outturns one quarter ahead, falling to 70% at eight
  quarters (`results/coverage_evaluation.json`). The 68% band covers 50% of
  GDP outturns at eight quarters. Scoring the band rescaling on held-out
  origins is in PR #19; the choice between labelling and dropping the 90%
  band is issue #12.
- **Monetary-policy level effects do not fade.** The posterior has a unit
  root: every draw checked on the test fixture has a largest companion
  eigenvalue above 1, and nothing truncates explosive draws. The GDP and
  CPI level responses to a monetary-policy shock stay away from zero out to
  80 quarters (PR #17 and its verification comment). Impulse-response
  magnitudes are not yet checked against the paper's Figures 2–3 (issue #11).
- **Year-on-year rates are log differences.** `forecast.py` computes YoY as
  a difference of 100·log levels, about x²/200 below the simple rates the
  BoE and OBR publish (0.04pp at 2.8%). The external comparison
  (`results/forecast_vs_boe_obr.md`) converts to the simple basis with
  `forecast.yoy_pct` (PR #18); `forecast_summary.md` and `fig9` remain log
  basis.

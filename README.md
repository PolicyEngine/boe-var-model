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
│   └── methodology.md      # method writeup + differences from the paper
├── src/boe_var/            # package (src layout)
│   ├── data.py             # load_data() -> transformed quarterly DataFrame
│   ├── bvar.py             # BVAR class: NIW posterior sampling
│   ├── identification.py   # zero/sign restrictions, draw_Q, identify
│   └── analysis.py         # IRFs, FEVDs, shocks, historical decompositions
├── scripts/
│   ├── download_data.py    # fetch raw series -> data/boe_var_data.csv
│   └── run_replication.py  # end-to-end: estimate, identify, write results/
├── data/                   # raw + assembled data (raw data git-ignored)
├── results/                # figures (fig2..fig6 *.png) and summary.md
├── tests/                  # unit tests for bvar and identification
├── RESOURCES.md            # annotated links: paper, code, methods, data
├── pyproject.toml          # package boe_var
└── requirements.txt
```

## Quickstart

Use the existing conda env `python313` (do not create a new venv):

```bash
conda activate python313
pip install -e .
python scripts/download_data.py     # builds data/boe_var_data.csv
python scripts/run_replication.py   # writes figures + summary to results/
```

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
- Fewer accepted identification draws by default (1,000+ vs the paper's
  10,000), and a simplified pandemic-prior treatment (Covid dummies as
  exogenous regressors). See `docs/methodology.md` for the full list of
  differences.

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

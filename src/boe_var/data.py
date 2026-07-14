"""Load the prepared quarterly dataset for the BoE SVAR replication."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "world_gdp",
    "world_cpi",
    "oil_price",
    "bank_rate",
    "eri",
    "cpisa",
    "cpi_energy",
    "uk_gdp",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _first_existing(repo_path: Path, packaged_path: Path) -> Path:
    """Prefer the repo-root copy (development checkout), else the packaged one.

    In a git checkout, `data/` and `results/` at the repo root are canonical
    (rebuilt by the scripts in `scripts/`); the copies under `boe_var/_data`
    and `boe_var/_results` are synchronised snapshots shipped as package data
    so a plain `pip install` works without cloning the repo.
    """
    return repo_path if repo_path.exists() else packaged_path


_DATA_PATH = _first_existing(
    _REPO_ROOT / "data" / "boe_var_data.csv",
    Path(__file__).resolve().parent / "_data" / "boe_var_data.csv",
)


def results_dir() -> Path:
    """Directory holding summary.md / forecast_summary.md.

    Repo-root `results/` in a development checkout; the packaged snapshot
    under `boe_var/_results` in a pip install.
    """
    repo_results = _REPO_ROOT / "results"
    if (repo_results / "summary.md").exists():
        return repo_results
    return Path(__file__).resolve().parent / "_results"


def results_path(name: str) -> Path:
    """Path to a committed results file (e.g. 'summary.md')."""
    return results_dir() / name


def load_data(path: str | Path = _DATA_PATH) -> pd.DataFrame:
    """Return the 8-variable quarterly dataset, already transformed.

    Index is a PeriodIndex (freq 'Q'); columns are ordered per SPEC.md and are
    100*log levels except `bank_rate` (percent level). Build the CSV with
    `python scripts/download_data.py`.
    """
    df = pd.read_csv(path)
    df.index = pd.PeriodIndex(df.pop("quarter"), freq="Q")
    df.index.name = "quarter"
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing columns: {missing}")
    return df[COLUMNS].astype(float)

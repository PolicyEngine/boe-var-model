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

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "boe_var_data.csv"


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

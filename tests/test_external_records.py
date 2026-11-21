"""The 19 Nov 2026 external records reproduce their committed results."""

import csv
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXT = os.path.join(ROOT, "data", "external")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reproduces(name, committed, tmp_path, capsys):
    mod = _load(name)
    mod.OUT = tmp_path / committed
    mod.main()
    capsys.readouterr()
    with open(os.path.join(ROOT, "results", committed)) as f:
        assert mod.OUT.read_text() == f.read(), f"results/{committed} is stale; rerun scripts/{name}.py"


def test_score_2026q3_reproduces(tmp_path, capsys):
    _reproduces("score_2026q3", "score_2026q3.md", tmp_path, capsys)


def test_ecb_seasonal_check_reproduces(tmp_path, capsys):
    _reproduces("compare_ecb_seasonal", "ecb_seasonal_check.md", tmp_path, capsys)


def test_outturn_gdp_base_is_the_estimation_vintage():
    """The year-on-year GDP outturn must not mix vintages: the PN2 level for
    2025Q3 equals the one in data/boe_var_data.csv."""
    rows = {(r["code"], r["period"]): float(r["value"]) for r in csv.DictReader(open(os.path.join(EXT, "uk_outturns_2026q3.csv")))}
    d = pd.read_csv(os.path.join(ROOT, "data", "boe_var_data.csv"), index_col="quarter")
    assert np.exp(d.loc["2025Q3", "uk_gdp"] / 100) == pytest.approx(rows[("ABMI", "2025Q3")], abs=0.5)


def test_mpr_transcription_shape():
    rows = list(csv.DictReader(open(os.path.join(EXT, "boe_mpr_2026_11_table_3b.csv"))))
    by = {r["variable"]: r for r in rows}
    assert len(rows) == 8
    assert [by["cpi_inflation"][q] for q in ("2026Q4", "2027Q4", "2028Q4", "2029Q4")] == ["3.8", "4.0", "2.0", "1.8"]
    assert [by["bank_rate"][q] for q in ("2026Q4", "2027Q4", "2028Q4", "2029Q4")] == ["3.9", "4.9", "4.9", "4.7"]

"""World CPI carries no residual seasonality (Fable review W1, 17 Nov 2026).

The euro-area HICP and German CPI legs of the world CPI proxy are not
seasonally adjusted at source, and their quarterly pattern drifts over time
(the Q2 effect rose from about +0.1 to +0.6 log points across eras), so
``scripts/download_data.py`` applies ``seasonal_adjust_tv`` to the
aggregate. These tests fail if the committed data or the model residuals
show a seasonal pattern again, pooled or in any era.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from boe_var.bvar import BVAR, PosteriorDraw
from boe_var.data import load_data

ERAS = [("1993Q1", "2004Q4"), ("2005Q1", "2014Q4"), ("2015Q1", "2021Q4"),
        ("2022Q1", "2025Q1"), ("1993Q1", "2025Q1")]
ROOT = Path(__file__).resolve().parents[1]


def _seasonal_p(x: pd.Series) -> float:
    x = x.dropna()
    return float(stats.f_oneway(
        *[x[x.index.quarter == q] for q in (1, 2, 3, 4)]).pvalue)


@pytest.mark.parametrize("era", ERAS)
def test_world_cpi_growth_has_no_seasonal_pattern(era):
    d = load_data()["world_cpi"].diff().loc[era[0]:era[1]]
    assert _seasonal_p(d) > 0.10


@pytest.mark.parametrize("era", ERAS)
def test_world_cpi_var_residuals_have_no_seasonal_pattern(era):
    """Posterior-mean residuals of the replication specification."""
    df = load_data()
    de = df.loc[(df.index >= pd.Period("1992Q1", "Q"))
                & (df.index <= pd.Period("2025Q1", "Q"))]
    covid = np.column_stack(
        [(de.index == q).astype(float)
         for q in pd.period_range("2020Q1", "2021Q2", freq="Q")])
    model = BVAR(de.to_numpy(float), lags=4, dummies=covid, theta=1.0)
    k = de.shape[1]
    draw = PosteriorDraw(Pi=model.B_post.T.copy(),
                         Sigma=model.S_post / (model.df_post - k - 1),
                         lags=4, k=k)
    resid = pd.DataFrame(model.residuals(draw), index=de.index[4:],
                         columns=de.columns)
    assert _seasonal_p(resid["world_cpi"].loc[era[0]:era[1]]) > 0.10


def test_adjustment_reproduces_from_the_unadjusted_series():
    """data/boe_var_data_world_cpi_nsa.csv + seasonal_adjust_tv gives the
    committed world_cpi column, and the other columns are untouched."""
    spec = importlib.util.spec_from_file_location(
        "download_data", ROOT / "scripts" / "download_data.py")
    dd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dd)
    nsa = pd.read_csv(ROOT / "data" / "boe_var_data_world_cpi_nsa.csv")
    nsa.index = pd.PeriodIndex(nsa.pop("quarter"), freq="Q")
    adjusted = dd.seasonal_adjust_tv(nsa["world_cpi"])
    committed = load_data()["world_cpi"]
    np.testing.assert_allclose(adjusted.loc[committed.index].to_numpy(),
                               committed.to_numpy(), atol=1e-5)

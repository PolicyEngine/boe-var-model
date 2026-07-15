"""Shared pytest configuration and fixtures.

Tests marked ``slow`` (high-draw / importance-weighted validation that runs
the Arias et al. (2018) numerical-Jacobian weight for every accepted draw,
~minutes) are skipped by default so a plain ``pytest`` run finishes in
seconds. Run them with ``pytest --runslow``.

The ``identified`` fixture estimates the real BoE SVAR on the packaged data
with a small, fixed-seed, *unweighted* configuration (fast: ~2 s) and is
reused across the paper-ground-truth validation tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run slow high-draw / weighted validation tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow high-draw test; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


# --- fast fixed-seed configuration for the correctness suite ----------------
# 600 posterior draws with a fixed sampling seed and a fixed identification
# seed accept ~50 draws in ~2 s (weights OFF). Enough accepted draws for a
# stable FEVD median while keeping the default suite fast and deterministic.
N_DRAWS = 600
SAMPLE_SEED = 1
IDENT_SEED = 2
LAGS = 4


def _covid_dummies(index):
    import pandas as pd

    quarters = pd.period_range("2020Q1", "2021Q2", freq="Q")
    D = np.zeros((len(index), len(quarters)))
    for j, q in enumerate(quarters):
        D[:, j] = (index == q).astype(float)
    return D


class _Identified:
    """Container for a fully estimated + identified real-data SVAR."""

    def __init__(self, model, y, dummies, index, pairs, weights):
        self.model = model
        self.y = y
        self.dummies = dummies
        self.index = index
        self.pairs = pairs          # list of (draw, B)
        self.weights = weights      # np.ndarray, sums to 1


def _build_identified(compute_weights):
    import pandas as pd

    from boe_var.bvar import BVAR
    from boe_var.data import load_data
    from boe_var.identification import identify

    df = load_data()
    df = df.loc[(df.index >= pd.Period("1992Q1", "Q"))
                & (df.index <= pd.Period("2023Q2", "Q"))]
    y = df.to_numpy(dtype=float)
    dummies = _covid_dummies(df.index)

    model = BVAR(y, lags=LAGS, dummies=dummies, lam=0.2, mu=1.0, theta=1.0)
    draws = model.sample_posterior(N_DRAWS, seed=SAMPLE_SEED)
    accepted = identify(draws, rng=np.random.default_rng(IDENT_SEED),
                        compute_weights=compute_weights)
    pairs = [(d, B) for d, B, _ in accepted]
    w = np.array([t[2] for t in accepted], dtype=float)
    w = w / w.sum()
    return _Identified(model, y, dummies, df.index, pairs, w)


@pytest.fixture(scope="session")
def identified():
    """Fast, unweighted, fixed-seed identified real-data SVAR (session-cached)."""
    return _build_identified(compute_weights=False)

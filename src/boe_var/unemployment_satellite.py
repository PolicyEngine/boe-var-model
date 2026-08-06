"""Unemployment satellite: UK unemployment from the model's GDP growth path.

This module is deliberately OUTSIDE the replication core. The 8-variable
sign-identified BVAR (identification.py) is untouched; the satellite maps
its UK GDP growth forecast into an unemployment-rate path through an estimated
GDP-growth-to-unemployment mapping -- a small dynamic regression of the
change in unemployment on GDP growth and its own lag, estimated by OLS
(1992Q1-2025Q1, furlough quarters dummied out):

    du_t = alpha + beta * g_t + gamma * du_{t-1} + eps_t

where ``u`` is the ONS unemployment rate (MGSX, quarterly, seasonally
adjusted, % aged 16+), ``du_t = u_t - u_{t-1}``, and ``g_t`` is year-on-year
real GDP growth in percent (the same units the forecast bands report).

Data provenance: ``_data/uk_unemployment_mgsx.csv`` is the ONS MGSX series
copied from the PolicyEngine Macro committed vintage store
(data/latest/uk_unemployment_rate.json, release 2026-07-20, spanning
1971Q1-2026Q1). The default estimation sample matches the VAR's:
1992Q1-2025Q1.

Forecast quantile mapping: conditional on the fitted coefficients the
unemployment path is a deterministic, strictly monotone DECREASING function
of each GDP growth input (beta < 0 on this sample). Feeding the GDP
median/lo68/hi68/lo90/hi90 paths through the mapping therefore yields the
unemployment median and bands with the bounds SWAPPED (high GDP growth ->
low unemployment). This transforms pointwise quantiles of the GDP path, not
the joint predictive distribution, and adds no satellite residual uncertainty --
the reported bands are therefore a lower bound on the true unemployment
uncertainty, and results must say so.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MGSX_PATH = Path(__file__).parent / "_data" / "uk_unemployment_mgsx.csv"

DEFAULT_SAMPLE = ("1992Q1", "2025Q1")

# Furlough decoupled GDP from unemployment: 2020 saw ~-20% y/y GDP prints
# with the unemployment rate nearly flat. Including these quarters
# attenuates the growth coefficient by ~3x (beta -0.047 -> -0.016 on the
# default sample), so they are dummied out of the fit by default. This is
# real-time feasible: at any origin from 2020Q1 on, the pandemic quarters
# already observed are known to be pandemic quarters.
COVID_QUARTERS = frozenset(
    {"2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2"}
)


def load_unemployment(path: Path = _MGSX_PATH) -> tuple[list[str], np.ndarray]:
    """Return (quarters, rates) for the packaged ONS MGSX series."""
    quarters, values = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            quarters.append(row["quarter"])
            values.append(float(row["unemployment_rate"]))
    return quarters, np.asarray(values, dtype=float)


@dataclass
class SatelliteFit:
    alpha: float
    beta: float
    gamma: float
    sigma: float          # residual std deviation, percentage points
    nobs: int
    sample: tuple[str, str]
    r2: float

    def forecast(self, g_path: np.ndarray, u_last: float,
                 du_last: float) -> np.ndarray:
        """Iterate the fitted equation over a GDP growth path (percent y/y).

        Deterministic: no shock draws. Monotone decreasing in every element
        of ``g_path`` when ``beta < 0``.
        """
        g_path = np.asarray(g_path, dtype=float)
        out = np.empty(g_path.shape[0])
        u, du = float(u_last), float(du_last)
        for h, g in enumerate(g_path):
            du = self.alpha + self.beta * float(g) + self.gamma * du
            u = u + du
            out[h] = u
        return out


def fit_satellite(g: np.ndarray, du: np.ndarray, du_lag: np.ndarray,
             sample_label: tuple[str, str] = DEFAULT_SAMPLE) -> SatelliteFit:
    """OLS of du on [1, g, du_lag]. All arrays aligned, same length."""
    g = np.asarray(g, dtype=float)
    du = np.asarray(du, dtype=float)
    du_lag = np.asarray(du_lag, dtype=float)
    if not (g.shape == du.shape == du_lag.shape):
        raise ValueError("g, du, du_lag must be aligned 1-D arrays")
    X = np.column_stack([np.ones_like(g), g, du_lag])
    coef, *_ = np.linalg.lstsq(X, du, rcond=None)
    resid = du - X @ coef
    dof = max(len(du) - 3, 1)
    ss_tot = float(((du - du.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return SatelliteFit(alpha=float(coef[0]), beta=float(coef[1]),
                   gamma=float(coef[2]),
                   sigma=float(np.sqrt((resid ** 2).sum() / dof)),
                   nobs=len(du), sample=sample_label, r2=r2)


def align_satellite_inputs(u_quarters: list[str], u: np.ndarray,
                      g_quarters: list[str], g: np.ndarray,
                      start: str, end: str,
                      exclude: frozenset[str] = COVID_QUARTERS):
    """Align du_t, g_t, du_{t-1} over [start, end] (inclusive, quarters).

    Quarters in ``exclude`` are dropped from the fit sample (see
    COVID_QUARTERS). Pass ``frozenset()`` to keep everything.
    """
    u_index = {q: i for i, q in enumerate(u_quarters)}
    rows = []
    for j, q in enumerate(g_quarters):
        if not (start <= q <= end) or q in exclude:
            continue
        i = u_index.get(q)
        if i is None or i < 2:
            continue
        rows.append((u[i] - u[i - 1], g[j], u[i - 1] - u[i - 2]))
    if len(rows) < 20:
        raise ValueError("too few aligned observations for a satellite fit")
    arr = np.asarray(rows, dtype=float)
    return arr[:, 1], arr[:, 0], arr[:, 2]  # g, du, du_lag


def fit_default(g_quarters: list[str], g_yoy: np.ndarray,
                start: str = DEFAULT_SAMPLE[0],
                end: str = DEFAULT_SAMPLE[1],
                exclude: frozenset[str] = COVID_QUARTERS) -> SatelliteFit:
    """Fit on the packaged MGSX series against a supplied GDP growth series."""
    uq, u = load_unemployment()
    g_a, du, du_lag = align_satellite_inputs(uq, u, g_quarters, g_yoy, start, end,
                                        exclude=exclude)
    return fit_satellite(g_a, du, du_lag, sample_label=(start, end))


def unemployment_bands(fit: SatelliteFit, gdp_bands: dict,
                       u_last: float, du_last: float) -> dict:
    """Map GDP growth bands (median/lo68/hi68/lo90/hi90 paths, percent y/y)
    to unemployment-rate bands. Bounds swap because beta < 0.

    The result carries only GDP-path uncertainty; satellite residual uncertainty
    is not sampled (see module docstring).
    """
    if fit.beta >= 0:
        raise ValueError("mapping assumes a negative growth coefficient")
    def f(path):
        return fit.forecast(np.asarray(path, dtype=float), u_last, du_last)
    return {
        "median": f(gdp_bands["median"]),
        "lo68": f(gdp_bands["hi68"]),
        "hi68": f(gdp_bands["lo68"]),
        "lo90": f(gdp_bands["hi90"]),
        "hi90": f(gdp_bands["lo90"]),
    }

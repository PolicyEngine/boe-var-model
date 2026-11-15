"""Recompute the tables in docs/band_decision.md from a coverage record.

Usage (repo root):

    PYTHONPATH=src python scripts/run_coverage_evaluation.py \\
        --output /tmp/cov.json --dump-record /tmp/record.npz
    PYTHONPATH=src python scripts/band_decision_tables.py /tmp/record.npz

Uses the production-specification arrays of the record: a cell is inside the
band when its stretch is at most 1, and the PIT gives the side of a miss.
The calibrated row applies the rolling-origin Gaussian factor from
``run_coverage_evaluation._scale_factor`` with at least 12 past outturns.
The bootstrap resamples blocks of consecutive origins.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from boe_var.data import COLUMNS, load_data

ROOT = Path(__file__).resolve().parent.parent
FIRST_ORIGIN = 80
MIN_TRAIN = 12
WINDOWS = [("2012Q2", "2014Q2"), ("2014Q3", "2016Q4"), ("2017Q1", "2019Q4"),
           ("2020Q1", "2021Q2"), ("2021Q3", "2023Q4"), ("2024Q1", None)]
EPISODES = (1, 3, 4)
CPI = ("world_cpi", "cpisa", "cpi_energy")


def _coverage_module():
    path = ROOT / "scripts" / "run_coverage_evaluation.py"
    spec = importlib.util.spec_from_file_location("rc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rolled(S, level, scale_factor):
    """Rolling-origin Gaussian-k inside indicator (NaN before MIN_TRAIN)."""
    n_o, H, k = S.shape
    out = np.full(S.shape, np.nan)
    for h in range(H):
        for j in range(k):
            inside = S[:, h, j] <= 1
            for i in range(n_o):
                n_train = i - h  # outturns observed by origin i
                if n_train >= MIN_TRAIN:
                    kf = scale_factor(level, float(inside[:n_train].mean()))
                    out[i, h, j] = S[i, h, j] <= (kf or 1.0)
    return out


def block_bootstrap(A, block, reps=10_000, seed=0):
    """Moving-block bootstrap over origins of pooled coverage."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    flat = A.reshape(n, -1)
    num = np.nansum(flat, axis=1)
    den = np.sum(~np.isnan(flat), axis=1)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(reps)
    for r in range(reps):
        starts = rng.integers(0, n - block + 1, n_blocks)
        ii = (starts[:, None] + np.arange(block)[None]).ravel()[:n]
        out[r] = num[ii].sum() / max(den[ii].sum(), 1)
    return out.std(), np.percentile(out, [2.5, 97.5])


def main(record_path: str) -> None:
    rc = _coverage_module()
    d = np.load(record_path)
    idx = load_data().index
    S90, S68, pit = d["prod_stretch90"], d["prod_stretch68"], d["prod_pit"]
    n_o, H, k = S90.shape
    tq = np.array([[idx[FIRST_ORIGIN + i + 1 + h] for h in range(H)]
                   for i in range(n_o)])
    miss = S90 > 1
    above = miss & (pit >= 0.5)
    below = miss & (pit < 0.5)

    def in_window(a, b):
        b = b or str(idx[-1])
        m = (tq >= pd.Period(a, "Q")) & (tq <= pd.Period(b, "Q"))
        return np.broadcast_to(m[:, :, None], miss.shape)

    print(f"origins {idx[FIRST_ORIGIN]}-{idx[FIRST_ORIGIN + n_o - 1]} ({n_o}), "
          f"cells {miss.size}, 90% misses {int(miss.sum())}, "
          f"raw 90% {1 - miss.mean():.4f}, raw 68% {(S68 <= 1).mean():.4f}")
    print("| Target quarters | Cells | 90% coverage | Share of misses "
          "| Above / below |")
    for a, b in WINDOWS:
        M = in_window(a, b)
        print(f"| {a}-{b or idx[-1]} | {int(M.sum())} | "
              f"{1 - miss[M].mean():.3f} | {miss[M].sum() / miss.sum():.0%} | "
              f"{int(above[M].sum())} / {int(below[M].sum())} |")
    E3 = np.zeros(miss.shape, dtype=bool)
    for w in EPISODES:
        E3 |= in_window(*WINDOWS[w])
    print(f"share of misses in the three episodes "
          f"{miss[E3].sum() / miss.sum():.0%}; "
          f"coverage outside them {1 - miss[~E3].mean():.3f}")

    y2022 = in_window("2022Q1", "2022Q4")
    print(f"dropping 2022 targets: {1 - miss[~y2022].mean():.3f}; dropping "
          f"2021Q3-2023Q4: {1 - miss[~in_window(*WINDOWS[4])].mean():.3f}")
    for w in (4, 1, 3):
        M = in_window(*WINDOWS[w])
        by_var = ", ".join(
            f"{c} {1 - miss[:, :, j][M[:, :, j]].mean():.2f}"
            for j, c in enumerate(COLUMNS))
        print(f"{WINDOWS[w][0]}-{WINDOWS[w][1]} by variable: {by_var}")
    infl = in_window(*WINDOWS[4])
    trio = [COLUMNS.index(c) for c in ("world_cpi", "cpisa", "bank_rate")]
    m3 = miss[:, :, trio][infl[:, :, trio]]
    a3 = above[:, :, trio][infl[:, :, trio]]
    print(f"world_cpi/cpisa/bank_rate misses in 2021Q3-2023Q4: "
          f"{int(m3.sum())}, of which above {int(a3.sum())}")

    cpi = [COLUMNS.index(c) for c in CPI]
    other = [j for j in range(k) if j not in cpi]
    print(f"CPI series share of misses "
          f"{miss[:, :, cpi].sum() / miss.sum():.0%}; coverage CPI "
          f"{1 - miss[:, :, cpi].mean():.3f}, others "
          f"{1 - miss[:, :, other].mean():.3f}")
    for h in (0, H - 1):
        print(f"CPI series h{h + 1}: below rate "
              f"{below[:, h, cpi].mean():.2f}, above rate "
              f"{above[:, h, cpi].mean():.2f}")
    print(f"h1: coverage {1 - miss[:, 0].mean():.3f}, "
          f"above {int(above[:, 0].sum())}, below {int(below[:, 0].sum())}")
    print(f"miss followed by miss at the next origin: "
          f"{(miss[:-1] & miss[1:]).sum() / miss[:-1].sum():.0%}; "
          f"unconditional {miss.mean():.0%}")
    print(f"PIT tails: above 95th {(pit > 0.95).mean():.3f}, "
          f"below 5th {(pit < 0.05).mean():.3f}")

    R90 = rolled(S90, 90, rc._scale_factor)
    scored = np.where(~np.isnan(R90).all(axis=(1, 2)))[0]
    per_cell = (~np.isnan(R90)).sum(axis=0)
    print(f"Gaussian k rolling origin 90%: {np.nanmean(R90):.4f} "
          f"(origins scored from {scored[0]}; {per_cell.min()}-"
          f"{per_cell.max()} scored origins per cell)")
    for name, A in (("raw 90%", (S90 <= 1).astype(float)),
                    ("Gaussian k rolling 90%", R90),
                    ("raw 68%", (S68 <= 1).astype(float))):
        for block in (4, 8, 12):
            se, ci = block_bootstrap(A, block)
            print(f"{name}: block {block}: s.e. {se:.3f}, "
                  f"95% interval {ci[0]:.2f}-{ci[1]:.2f}")
    per_origin = (S90 <= 1).reshape(n_o, -1).mean(1)
    r1 = np.corrcoef(per_origin[:-1], per_origin[1:])[0, 1]
    print(f"per-origin 90% coverage lag-1 autocorrelation {r1:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])

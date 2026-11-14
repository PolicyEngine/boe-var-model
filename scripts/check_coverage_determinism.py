"""Does this machine reproduce the committed coverage artefact? (issue #26)

Reruns ``scripts/run_coverage_evaluation.py`` at the committed configuration
and compares the two ``coverage`` blocks (headline and production
specification, 128 cells each) with ``results/coverage_evaluation.json``.
Exits 1 if any cell differs.

It also prints what the comparison depends on: the numpy, scipy and BLAS
builds, the CPU, and a fingerprint of the two factorisations the random paths
go through. ``forecast.sample_forecast`` builds its shock factor from
``np.linalg.eigh``, whose eigenvector signs are not fixed by the algorithm, so
a different LAPACK build can return a different (equally valid) factor and
hence different shock paths from the same seed.

Run:  python scripts/check_coverage_determinism.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy

from boe_var import forecast as fc
from boe_var.bvar import BVAR
from boe_var.data import load_data

ROOT = Path(__file__).resolve().parent.parent
COMMITTED = ROOT / "results" / "coverage_evaluation.json"
SCRIPT = ROOT / "scripts" / "run_coverage_evaluation.py"


def environment() -> dict:
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(line.split(":", 1)[1].strip()
                       for line in fh if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": f"{blas.get('name')} {blas.get('version')}",
    }


def fingerprint() -> dict:
    """Signs and values of one draw's factorisation and one shock path."""
    y = load_data().to_numpy()[:81]
    draw = BVAR(y, lags=4).sample_posterior(1, seed=0)[0]
    sigma = np.asarray(draw.Sigma, dtype=float)
    _, vecs = np.linalg.eigh(0.5 * (sigma + sigma.T))
    path = fc.sample_forecast(draw, y, horizons=2,
                              rng=np.random.default_rng(0))
    return {
        "sigma_00": repr(float(sigma[0, 0])),
        "eigenvector_first_row_signs": "".join(
            "+" if v >= 0 else "-" for v in vecs[0]),
        "path_h1": [repr(float(v)) for v in path[0, :3]],
    }


def cells(block: dict) -> dict:
    return {(lv, h, var): val
            for lv, by_h in block.items()
            for h, by_var in by_h.items()
            for var, val in by_var.items()}


def main() -> int:
    print(json.dumps({"environment": environment(),
                      "fingerprint": fingerprint()}, indent=1))
    committed = json.loads(COMMITTED.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "coverage.json"
        subprocess.run(
            [sys.executable, str(SCRIPT), "--output", str(out),
             "--seed", str(committed["seed"]),
             "--draws", str(committed["draws_per_origin"]),
             "--paths", str(committed["paths_per_draw"])],
            check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        fresh = json.loads(out.read_text())

    failed = False
    for name, pick in (("headline", lambda r: r["coverage"]),
                       ("production_spec",
                        lambda r: r["production_spec"]["coverage"])):
        old, new = cells(pick(committed)), cells(pick(fresh))
        diff = {k: (old[k], new[k]) for k in old if old[k] != new[k]}
        worst = max((abs(a - b) for a, b in diff.values()), default=0.0)
        print(f"{name}: {len(diff)} of {len(old)} cells differ "
              f"(largest difference {worst:.4f})")
        for key, (a, b) in sorted(diff.items())[:10]:
            print(f"  {'/'.join(key)}: committed {a}, this run {b}")
        failed = failed or bool(diff)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

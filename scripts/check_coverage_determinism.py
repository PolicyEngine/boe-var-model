"""Does this machine reproduce the committed coverage artefact? (issue #26)

Reruns ``scripts/run_coverage_evaluation.py`` at the committed configuration
and compares the coverage blocks (levels and year on year, headline and
production specification, 128 cells each) with
``results/coverage_evaluation.json``. Exits 1 if any cell differs.

It also prints what the comparison depends on: the numpy, scipy and BLAS
builds, the CPU, and a fingerprint of one draw's shock factor and path.
Before #26 the shock factor came from ``np.linalg.eigh``, whose eigenvector
signs depend on the LAPACK build and the CPU kernel, so the same seed gave
different cells on different machines. ``forecast.shock_factor`` now uses
the Cholesky factor, which is unique; this check is how that is confirmed on
a new machine.

Run:  python scripts/check_coverage_determinism.py [--save PATH]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import runpy
import shutil
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
BLOCKS = {
    "headline": lambda r: r["coverage"],
    "headline_yoy": lambda r: r["coverage_yoy"],
    "production_spec": lambda r: r["production_spec"]["coverage"],
    "production_spec_yoy": lambda r: r["production_spec"]["coverage_yoy"],
}


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
    """One draw's Sigma, shock factor and first path step."""
    y = load_data().to_numpy()[:81]
    draw = BVAR(y, lags=4).sample_posterior(1, seed=0)[0]
    factor = fc.shock_factor(draw.Sigma)
    path = fc.sample_forecast(draw, y, horizons=2,
                              rng=np.random.default_rng(0))
    return {
        "sigma_00": repr(float(draw.Sigma[0, 0])),
        "factor_diagonal_signs": "".join(
            "+" if v >= 0 else "-" for v in np.diag(factor)),
        "path_h1": [repr(float(v)) for v in path[0, :3]],
    }


def regenerate(out: Path, committed: dict) -> dict:
    """Run scripts/run_coverage_evaluation.py in this process."""
    argv = sys.argv
    sys.argv = [str(SCRIPT), "--output", str(out),
                "--seed", str(committed["seed"]),
                "--draws", str(committed["draws_per_origin"]),
                "--paths", str(committed["paths_per_draw"])]
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        sys.argv = argv
    return json.loads(out.read_text())


def cells(block: dict) -> dict:
    return {(lv, h, var): val
            for lv, by_h in block.items()
            for h, by_var in by_h.items()
            for var, val in by_var.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", type=Path,
                    help="also write the regenerated file here")
    args = ap.parse_args()
    print(json.dumps({"environment": environment(),
                      "fingerprint": fingerprint()}, indent=1))
    committed = json.loads(COMMITTED.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "coverage.json"
        fresh = regenerate(out, committed)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(out, args.save)

    failed = False
    for name, pick in BLOCKS.items():
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

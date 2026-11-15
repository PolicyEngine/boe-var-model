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

Run:  python scripts/check_coverage_determinism.py [--save PATH]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from scipy.linalg import cholesky

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
    many = BVAR(y, lags=4).sample_posterior(500, seed=1)
    signs = "".join(
        "".join("+" if v >= 0 else "-" for v in np.linalg.eigh(
            0.5 * (d.Sigma + d.Sigma.T))[1][0]) for d in many)
    return {
        "eigenvector_signs_500_draws_sha256": hashlib.sha256(
            signs.encode()).hexdigest()[:16],
        "sigma_00": repr(float(sigma[0, 0])),
        "eigenvector_first_row_signs": "".join(
            "+" if v >= 0 else "-" for v in vecs[0]),
        "path_h1": [repr(float(v)) for v in path[0, :3]],
    }


def cholesky_sample_forecast(draw, y_hist, dummies_hist=None,
                             horizons: int = 13, rng=None):
    """``forecast.sample_forecast`` with a Cholesky shock factor.

    Same distribution as the eigen factor, but the factor is unique (positive
    diagonal), so it does not depend on the LAPACK build. Used only with
    ``--factor cholesky`` to test the fix proposed in #26.
    """
    del dummies_hist
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.asarray(draw.Sigma, dtype=float)
    ls = cholesky(0.5 * (sigma + sigma.T), lower=True)
    shocks = rng.standard_normal((horizons, draw.k)) @ ls.T
    return fc._iterate(draw, y_hist, horizons, shocks=shocks)


def regenerate(out: Path, committed: dict, factor: str) -> dict:
    """Run scripts/run_coverage_evaluation.py in this process."""
    if factor == "cholesky":
        fc.sample_forecast = cholesky_sample_forecast
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
    ap.add_argument("--factor", choices=("eigh", "cholesky"), default="eigh",
                    help="shock factor for the random paths; cholesky tests "
                         "the fix proposed in #26 and is expected to differ "
                         "from the committed (eigh) file")
    args = ap.parse_args()
    print(json.dumps({"factor": args.factor,
                      "environment": environment(),
                      "fingerprint": fingerprint()}, indent=1))
    committed = json.loads(COMMITTED.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "coverage.json"
        fresh = regenerate(out, committed, args.factor)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(out, args.save)

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

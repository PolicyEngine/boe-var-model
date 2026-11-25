"""Diagnostic for the runner-dependent stochastic-volatility tests (#12).

Hypothesis: the SV Gibbs sampler is chaotic (discrete mixture-indicator
draws), and BLAS results can differ in the last bit with the memory alignment
of the input array, so the same data in two differently aligned buffers can
give different chains on some CPUs. The no-leakage tests compare a view of
the dataset with a fresh copy of it, i.e. two alignments.

Prints, for byte offsets 0..56 of the same data inside a larger buffer, the
largest difference from the offset-0 run for (a) the homoskedastic BVAR and
(b) the common-SV sampler. Scratch only; not part of the package.
"""
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from boe_var import common_sv as sv  # noqa: E402
from boe_var.bvar import BVAR  # noqa: E402
from boe_var.data import load_data  # noqa: E402


def at_offset(a: np.ndarray, offset: int) -> np.ndarray:
    """The same values in a buffer whose data pointer is ``offset`` bytes
    past a 64-byte boundary."""
    raw = np.empty(a.nbytes + 128, dtype=np.uint8)
    start = (-raw.ctypes.data) % 64 + offset
    out = np.frombuffer(raw.data, dtype=a.dtype, count=a.size,
                        offset=start).reshape(a.shape)
    out[...] = a
    assert out.ctypes.data % 64 == offset % 64
    return out


def main():
    print(platform.platform(), platform.machine())
    try:
        print(subprocess.run("lscpu | grep -E 'Model name|Flags' | cut -c1-400",
                             shell=True, capture_output=True, text=True).stdout)
    except Exception:
        pass
    np.show_config()
    df = load_data()
    y = df.to_numpy(float)
    o = list(df.index).index(pd.Period("2021Q1", "Q"))
    train = np.ascontiguousarray(y[: o + 1])

    def bvar(a):
        d = BVAR(a, lags=4).sample_posterior(6, seed=4)
        return np.concatenate([np.ravel(x.Pi) for x in d])

    def svrun(a):
        m = sv.CommonSVBVAR(a, lags=4)
        d = m.sample(6, seed=4, burn=20, thin=1)
        return np.concatenate([np.ravel(x.Pi) for x in d]
                              + [np.array([x.h_T for x in d])])

    base_b, base_s = bvar(at_offset(train, 0)), svrun(at_offset(train, 0))
    worst_b = worst_s = 0.0
    for off in range(0, 64, 8):
        a = at_offset(train, off)
        db = float(np.abs(bvar(a) - base_b).max())
        ds = float(np.abs(svrun(a) - base_s).max())
        worst_b, worst_s = max(worst_b, db), max(worst_s, ds)
        print(f"offset {off:2d}: BVAR max diff {db:.3e}   SV max diff {ds:.3e}")
    # the tests' own pair: a view of the dataset against a fresh copy
    view = y[: o + 4]
    for i in range(5):
        copy = view.copy()
        ds = float(np.abs(svrun(copy[: o + 1]) - svrun(view[: o + 1])).max())
        print(f"view vs copy {i}: pointers mod 64 = {view.ctypes.data % 64}, "
              f"{copy.ctypes.data % 64}; SV max diff {ds:.3e}")
    # the sampler builds its own regressor arrays, whose alignment depends
    # on the state of the allocator; perturb the heap between identical runs
    rng = np.random.default_rng(0)
    keep, n_diff, worst_r, worst_rb = [], 0, 0.0, 0.0
    ref_s, ref_b = svrun(train), bvar(train)
    for i in range(40):
        keep.append(np.empty(int(rng.integers(1, 4000)), dtype=np.uint8))
        if i % 3 == 0 and keep:
            keep.pop(int(rng.integers(len(keep))))
        ds = float(np.abs(svrun(train) - ref_s).max())
        worst_rb = max(worst_rb, float(np.abs(bvar(train) - ref_b).max()))
        n_diff += ds > 0
        worst_r = max(worst_r, ds)
    print(f"REPEAT identical input, 40 runs with heap perturbation: "
          f"{n_diff} differ, worst SV {worst_r:.3e}, worst BVAR {worst_rb:.3e}")
    print(f"RESULT worst BVAR {worst_b:.3e} worst SV {worst_s:.3e}")


if __name__ == "__main__":
    main()

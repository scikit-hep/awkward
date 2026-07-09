# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Benchmark: kernel fusion vs. the per-node interpreter on the lazy IR.

Two things are measured for element-wise chains (and a filter pipeline) over
jagged arrays:

1. **Launch proxy** — how many separately-dispatched ops / materialized
   intermediates each path produces.  In the eager/interpreter model every IR
   op is its own launch + its own intermediate buffer; fusion collapses a
   whole element-wise region into one.  This count is backend-independent and
   is what predicts the GPU kernel-launch saving directly.

2. **Wall-clock (CPU)** — ``compute(fuse=False)`` (interpreter, one ``ak`` op
   per node) vs. ``compute(fuse=True)`` (one flat-buffer NumPy pass over the
   shared content).  The CPU win comes from paying the ``ak`` dispatch /
   broadcast cost once instead of N times; on a GPU the same collapse also
   removes N-1 kernel launches and keeps intermediates in registers/L1 (not
   measurable here — no GPU — but see the projection printed at the end).

Usage::

    python studies/cccl/bench_lazy_fusion.py                 # print table
    python studies/cccl/bench_lazy_fusion.py --out bench.json # + save JSON
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import awkward as ak

REPEATS = 7


def make_jagged(n_lists, max_len=6, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, max_len, size=n_lists)
    offsets = np.zeros(n_lists + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    content = rng.random(int(offsets[-1]))
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets), ak.contents.NumpyArray(content)
        )
    )


def build_chain(arr, depth):
    """A depth-op element-wise chain: (((a*1.001 + 0.5) * 1.001 + 0.5) ...)."""
    la = ak.cpu.lazy(arr)
    expr = la
    for _ in range(depth):
        expr = expr * 1.001 + 0.5
    return expr


def build_pipeline(arr):
    """scale -> shift -> filter-by-transformed-condition (a fan-out DAG)."""
    la = ak.cpu.lazy(arr)
    t = la * 2.0 + 1.0
    return t.filter(t > 1.5)


def _time(make, fuse):
    best = float("inf")
    for _ in range(REPEATS):
        expr = make()  # fresh graph each rep (avoid result caching)
        t0 = time.perf_counter()
        expr.compute(fuse=fuse)
        best = min(best, time.perf_counter() - t0)
    return best


def bench_case(name, make, expected_regions=None):
    # warm the op-compile caches
    make().compute(fuse=True)
    make().compute(fuse=False)

    stats = make().fusion_stats()
    interp = _time(make, fuse=False)
    fused = _time(make, fuse=True)
    row = {
        "case": name,
        "elementwise_ops": stats["elementwise_before"],
        "fused_regions": stats["fused_regions"],
        "interp_ms": interp * 1e3,
        "fused_ms": fused * 1e3,
        "speedup": interp / fused if fused else float("nan"),
    }
    return row


def run(n_lists):
    arr = make_jagged(n_lists)
    n_items = len(ak.flatten(arr))
    rows = []
    for depth in (1, 2, 4, 8, 16):
        rows.append(bench_case(f"chain d={depth:<2}", lambda d=depth: build_chain(arr, d)))
    rows.append(bench_case("filter pipeline", lambda: build_pipeline(arr)))
    return {"n_lists": n_lists, "n_items": int(n_items), "rows": rows}


def print_table(result):
    print(f"\n### {result['n_lists']:,} lists / {result['n_items']:,} items")
    hdr = f"{'case':<16}{'ew ops':>7}{'kernels':>9}{'interp ms':>11}{'fused ms':>10}{'speedup':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in result["rows"]:
        print(
            f"{r['case']:<16}{r['elementwise_ops']:>7}{r['fused_regions']:>9}"
            f"{r['interp_ms']:>11.2f}{r['fused_ms']:>10.2f}{r['speedup']:>8.2f}x"
        )


def gpu_projection(results, per_launch_us=6.0):
    """Rough GPU launch-overhead saving from the measured launch reduction.

    Uses a representative per-launch cost (Nsight traces in the follow-up plan
    show ~0.3-1.2 ms per eager op incl. compute; 6 us is a conservative
    *pure-launch* floor).  Illustrative only.
    """
    saved = 0
    for res in results:
        for r in res["rows"]:
            saved += r["elementwise_ops"] - r["fused_regions"]
    print(
        f"\nGPU projection: fusion removes {saved} kernel launches across the "
        f"cases above;\nat ~{per_launch_us:.0f} us/launch pure overhead that is "
        f"~{saved * per_launch_us / 1e3:.2f} ms saved before counting the "
        f"on-chip data-reuse win."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[200_000, 2_000_000],
        help="list counts to benchmark",
    )
    args = ap.parse_args()

    results = [run(n) for n in args.sizes]
    for res in results:
        print_table(res)
    gpu_projection(results)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Benchmark the advantages of the lazy IR over eager evaluation.

Scenarios (CPU backend, plus the same pipeline on CUDA when cupy is present):

  build            Building the expression graph is O(graph size), not
                   O(data): microseconds regardless of array length.
  branch_1of8      Eight derived pipelines are defined but only one is
                   materialized. Eager pays for all eight; lazy pays for one.
  shared_subexpr   Eight results share a 4-op common subexpression. The
                   executor memo computes the shared part once; eager
                   recomputes it per result.
  recompute        Asking for an already-computed result. Lazy returns the
                   memoized array; eager runs the pipeline again.
  fastpath_filter  The backend fast path (vectorized NumPy on cpu, CCCL
                   single-pass select on cuda) vs eager boolean mask + slice,
                   which materializes a jagged boolean intermediate.
  straight_line    Caveat, for honesty: a pipeline computed exactly once with
                   no sharing/reuse. Lazy adds only interpreter overhead here
                   and is expected to be ~equal to eager (slightly slower).

Each timing is best-of-REPEATS wall time.  The executor memo is LRU-bounded
(64 entries / 256 MiB by default), but workloads still use a fresh lazy
wrapper per repetition so repetitions cannot serve each other from the memo,
which would make the "lazy" times meaninglessly fast.

Usage:
    python bench_lazy_ir.py [--lists N] [--repeats R] [out.json]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import numpy as np

import awkward as ak


def make_array(num_lists, backend="cpu", seed=12345):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 20, num_lists)
    content = rng.normal(50, 25, int(counts.sum()))
    offsets = np.zeros(num_lists + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    arr = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets), ak.contents.NumpyArray(content)
        )
    )
    return ak.to_backend(arr, backend) if backend != "cpu" else arr


def best_of(f, repeats):
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_backend(backend, num_lists, repeats):
    arr = make_array(num_lists, backend)
    lazy = ak.cpu.lazy if backend == "cpu" else ak.cuda.lazy
    results = {}

    def row(name, lazy_s, eager_s, note=""):
        results[name] = {"lazy_s": lazy_s, "eager_s": eager_s}
        ratio = eager_s / lazy_s if lazy_s > 0 else float("inf")
        print(
            f"  {name:<16} lazy {lazy_s * 1e3:>10.3f} ms   "
            f"eager {eager_s * 1e3:>10.3f} ms   {ratio:>8.1f}x  {note}"
        )

    # ------------------------------------------------------------------ build
    la = lazy(arr)
    t_build = best_of(lambda: (la * 2 + 1).filter(la > 3), repeats)
    t_eager = best_of(lambda: (arr * 2 + 1)[arr > 3], repeats)
    row("build", t_build, t_eager, "(graph construction vs full evaluation)")

    # ----------------------------------------------------------- branch_1of8
    def lazy_one_branch():
        la = lazy(arr)
        branches = [(la * 2 + i).filter(la > i) for i in range(8)]
        return branches[3].compute()

    def eager_all_branches():
        branches = [(arr * 2 + i)[arr > i] for i in range(8)]
        return branches[3]

    row(
        "branch_1of8",
        best_of(lazy_one_branch, repeats),
        best_of(eager_all_branches, repeats),
        "(only 1 of 8 pipelines materialized)",
    )

    # -------------------------------------------------------- shared_subexpr
    def lazy_shared():
        la = lazy(arr)
        shared = (la * 2 + 1) * 3 - 2
        return [(shared + i).compute() for i in range(8)]

    def eager_shared():
        return [(((arr * 2 + 1) * 3 - 2) + i) for i in range(8)]

    row(
        "shared_subexpr",
        best_of(lazy_shared, repeats),
        best_of(eager_shared, repeats),
        "(4-op subexpression shared by 8 results)",
    )

    # ------------------------------------------------------------- recompute
    la = lazy(arr)
    res = (la * 2 + 1).filter(la > 3)
    res.compute()
    row(
        "recompute",
        best_of(res.compute, repeats),
        best_of(lambda: (arr * 2 + 1)[arr > 3], repeats),
        "(second request for the same result)",
    )

    # ------------------------------------------------------- fastpath_filter
    if backend == "cpu":
        from awkward._connect.cpu.helpers import filter_lists
    else:
        from awkward._connect.cuda.helpers import filter_lists

    row(
        "fastpath_filter",
        best_of(lambda: filter_lists(arr, lambda x: x > 60), repeats),
        best_of(lambda: arr[arr > 60], repeats),
        "(predicate fast path vs mask + slice)",
    )

    # --------------------------------------------------------- straight_line
    def lazy_once():
        la = lazy(arr)
        return (la * 2 + 1).filter(la > 3).compute()

    row(
        "straight_line",
        best_of(lazy_once, repeats),
        best_of(lambda: (arr * 2 + 1)[arr > 3], repeats),
        "(caveat: single-shot pipeline, expect ~1x)",
    )

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", help="optional JSON output path")
    parser.add_argument("--lists", type=int, default=500_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)

    backends = ["cpu"]
    try:
        import cupy  # noqa: F401

        backends.append("cuda")
    except ImportError:
        print("cupy not available; running CPU backend only\n")

    all_results = {}
    for backend in backends:
        n_elements = int(make_array(args.lists, "cpu").layout.offsets[-1])
        print(
            f"backend={backend}  lists={args.lists:,}  elements≈{n_elements:,}  "
            f"best of {args.repeats}"
        )
        all_results[backend] = bench_backend(backend, args.lists, args.repeats)
        print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

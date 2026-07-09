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


def _lazy(arr):
    """Wrap ``arr`` with the lazy entry point for its backend."""
    if arr.layout.backend.name == "cuda":
        return ak.cuda.lazy(arr)
    return ak.cpu.lazy(arr)


def build_chain(arr, depth):
    """A depth-op element-wise chain: (((a*1.001 + 0.5) * 1.001 + 0.5) ...)."""
    expr = _lazy(arr)
    for _ in range(depth):
        expr = expr * 1.001 + 0.5
    return expr


def build_pipeline(arr):
    """scale -> shift -> filter-by-transformed-condition (a fan-out DAG)."""
    t = _lazy(arr) * 2.0 + 1.0
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


# ----------------------------------------------------------------------
# GPU arm: measured with CUDA events, validated against the interpreter
# ----------------------------------------------------------------------


def _matches(a, b):
    """Fused vs interpreter agreement, tolerant to float reduction-order drift."""
    la = ak.to_backend(a, "cpu")
    lb = ak.to_backend(b, "cpu")
    try:
        fa = np.asarray(ak.to_numpy(ak.flatten(la, axis=None)))
        fb = np.asarray(ak.to_numpy(ak.flatten(lb, axis=None)))
    except Exception:  # noqa: BLE001 - fall back to structural compare
        return ak.to_list(la) == ak.to_list(lb)
    if fa.shape != fb.shape:
        return False
    if np.issubdtype(fa.dtype, np.floating):
        return bool(np.allclose(fa, fb, rtol=1e-9, atol=0.0))
    return bool(np.array_equal(fa, fb))


def _time_gpu(cp, make, fuse, reps):
    """ms/op on the device timeline (invalidate+recompute so kernels re-run).

    The graph is built once outside the timed window; each iteration clears the
    cached result (host-side, ~us) and recomputes, so only device work is
    timed.  Warms up first to exclude NVRTC / cuda.compute JIT.
    """
    expr = make()
    expr.compute(fuse=fuse)  # warmup: compile + op-cache fill
    cp.cuda.Device().synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(reps):
        expr.invalidate()
        expr.compute(fuse=fuse)
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / reps


def bench_case_gpu(name, make, cp, reps=50):
    # correctness + fusion-engagement check (once, untimed)
    fused_expr = make()
    fused_res = fused_expr.compute(fuse=True)
    hits = dict(fused_expr.executor.fused_hits)
    interp_res = make().compute(fuse=False)
    same = _matches(fused_res, interp_res)

    interp = _time_gpu(cp, make, fuse=False, reps=reps)
    fused = _time_gpu(cp, make, fuse=True, reps=reps)
    stats = make().fusion_stats()
    return {
        "case": name,
        "elementwise_ops": stats["elementwise_before"],
        "fused_regions": stats["fused_regions"],
        "cuda_kernel_hits": hits["cuda"],
        "matches_interpreter": bool(same),
        "interp_ms": interp,
        "fused_ms": fused,
        "speedup": interp / fused if fused else float("nan"),
    }


def build_chain_sum(arr, depth):
    """A depth-op element-wise chain reduced per sublist (transform+reduce)."""
    return build_chain(arr, depth).sum()


def _safe_case(name, make, cp):
    try:
        return bench_case_gpu(name, make, cp)
    except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
        return {"case": name, "error": f"{type(exc).__name__}: {exc}"}


def run_gpu(n_lists):
    import cupy as cp

    base = make_jagged(n_lists)
    n_items = int(np.asarray(base.layout.offsets)[-1])  # avoid reducers here
    arr = ak.to_backend(base, "cuda")
    rows = []
    for depth in (1, 2, 4, 8, 16):
        rows.append(
            _safe_case(f"chain d={depth:<2}", lambda d=depth: build_chain(arr, d), cp)
        )
    rows.append(_safe_case("chain d=4 ->sum", lambda: build_chain_sum(arr, 4), cp))
    rows.append(_safe_case("filter pipeline", lambda: build_pipeline(arr), cp))
    return {"device": "cuda", "n_lists": n_lists, "n_items": n_items, "rows": rows}


def print_table_gpu(result):
    print(f"\n### [GPU] {result['n_lists']:,} lists / {result['n_items']:,} items")
    hdr = (
        f"{'case':<16}{'ew ops':>7}{'kernels':>8}{'cuda hit':>9}{'match':>7}"
        f"{'interp ms':>11}{'fused ms':>10}{'speedup':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in result["rows"]:
        if "error" in r:
            print(f"{r['case']:<16}  ERROR: {r['error']}")
            continue
        print(
            f"{r['case']:<16}{r['elementwise_ops']:>7}{r['fused_regions']:>8}"
            f"{r['cuda_kernel_hits']:>9}{('yes' if r['matches_interpreter'] else 'NO!'):>7}"
            f"{r['interp_ms']:>11.3f}{r['fused_ms']:>10.3f}{r['speedup']:>8.2f}x"
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
    ap.add_argument(
        "--gpu",
        choices=("auto", "on", "off"),
        default="auto",
        help="run the CUDA arm ('auto' = on iff cupy imports)",
    )
    args = ap.parse_args()

    cpu_results = [run(n) for n in args.sizes]
    for res in cpu_results:
        print_table(res)

    gpu_results = []
    want_gpu = args.gpu != "off"
    if want_gpu:
        try:
            import cupy  # noqa: F401
        except ImportError:
            if args.gpu == "on":
                raise
            print("\n[GPU] cupy not available; skipping CUDA arm.")
            want_gpu = False
    if want_gpu:
        gpu_results = [run_gpu(n) for n in args.sizes]
        for res in gpu_results:
            print_table_gpu(res)
        # sanity: warn loudly if fusion never engaged on device
        for res in gpu_results:
            for r in res["rows"]:
                if "error" in r:
                    continue
                if r["fused_regions"] and r["cuda_kernel_hits"] == 0:
                    print(f"[GPU] WARNING: {r['case']} fell back (no cuda kernel).")
                if not r["matches_interpreter"]:
                    print(f"[GPU] WARNING: {r['case']} result != interpreter!")

    gpu_projection(cpu_results)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"cpu": cpu_results, "gpu": gpu_results}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

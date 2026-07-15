# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Scaling benchmarks + plots for the lazy IR: eager vs interpreter vs fused.

Measures, at 10K / 1M / 10M events (jagged outer lists, same generator as
``bench_lazy_fusion.py``):

1. **Runtime vs dataset size** — eager ``ak`` ops, ``compute(fuse=False)``
   (per-node interpreter), and ``compute(fuse=True)`` (fused kernels).
2. **GPU speedup** — CPU time / GPU time per mode and size (needs one JSON
   from each backend; with a single backend the plot falls back to
   fused-over-eager speedup on that backend).
3. **Kernel launch count** — graph-level, exact, from ``fusion_stats()``:
   ``launches_interp = elementwise_before + (materialized - fused_regions)``
   and ``launches_fused = materialized``.  Backend-independent.
4. **Memory traffic** — model: one read + one write of the flat content per
   launch (boolean intermediates counted at full width — conservative), i.e.
   ``launches x 2B``.  On CUDA a *measured* series is added: bytes requested
   from the CuPy memory pool during one compute (a MemoryHook counts every
   allocation the pipeline makes, pool-served or not).

Usage (one run per backend, then plot from the JSONs)::

    # on a CPU node / laptop
    python studies/cccl/bench_lazy_ir_plots.py --backend cpu  --out lazyir_cpu.json
    # on a GPU node (della)
    python studies/cccl/bench_lazy_ir_plots.py --backend cuda --out lazyir_cuda.json
    # plots (either one JSON or both)
    python studies/cccl/bench_lazy_ir_plots.py --plot lazyir_cpu.json lazyir_cuda.json

Writes into ``studies/cccl/figs/``:
    lazy_ir_runtime_vs_size.png, lazy_ir_gpu_speedup.png,
    lazy_ir_kernel_launches.png, lazy_ir_memory_traffic.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

import awkward as ak

SIZES = (10_000, 1_000_000, 10_000_000)
DEPTH = 8  # elementwise-chain length (the fusion headline case)
REPEATS = 5
FIGS = pathlib.Path(__file__).parent / "figs"

# palette matching make_slide_figs.py / the Marp deck
GREEN = "#76B900"
FAST = "#4c8c00"
SLOW = "#b23b2e"
INK = "#123f4d"
GREY = "#8a8a8a"


# ----------------------------------------------------------------------
# data + pipelines (same shapes as bench_lazy_fusion.py, for comparability)
# ----------------------------------------------------------------------


def make_jagged(n_lists, backend, max_len=6, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, max_len, size=n_lists)
    offsets = np.zeros(n_lists + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    content = rng.random(int(offsets[-1]))
    arr = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets), ak.contents.NumpyArray(content)
        )
    )
    return ak.to_backend(arr, backend) if backend != "cpu" else arr


def _lazy(arr):
    from awkward._connect import cpu, cuda

    if arr.layout.backend.name == "cuda":
        return cuda.lazy(arr)
    return cpu.lazy(arr)


def lazy_chain(arr, depth=DEPTH):
    expr = _lazy(arr)
    for _ in range(depth):
        expr = expr * 1.001 + 0.5
    return expr


def eager_chain(arr, depth=DEPTH):
    x = arr
    for _ in range(depth):
        x = x * 1.001 + 0.5
    return x


def lazy_filter_pipeline(arr, depth=DEPTH):  # depth unused; uniform signature
    t = _lazy(arr) * 2.0 + 1.0
    return t.filter(t > 1.5)


def eager_filter_pipeline(arr, depth=DEPTH):
    t = arr * 2.0 + 1.0
    return t[t > 1.5]


PIPELINES = {
    "chain": (lazy_chain, eager_chain),
    "filter": (lazy_filter_pipeline, eager_filter_pipeline),
}


# ----------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------


def _sync(backend):
    if backend == "cuda":
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()


def time_best(fn, backend, repeats=REPEATS):
    fn()  # warmup (op/JIT caches, pool growth)
    _sync(backend)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync(backend)
        best = min(best, time.perf_counter() - t0)
    return best


def measured_alloc_bytes(fn, backend):
    """Bytes requested from the CuPy pool during one call (CUDA only)."""
    if backend != "cuda":
        return None
    import cupy as cp

    class _Meter(cp.cuda.memory_hook.MemoryHook):
        name = "lazy_ir_alloc_meter"

        def __init__(self):
            self.bytes = 0

        def malloc_preprocess(self, **kwargs):
            self.bytes += kwargs["mem_size"]

    fn()  # warmup outside the meter
    _sync(backend)
    meter = _Meter()
    with meter:
        fn()
        _sync(backend)
    return int(meter.bytes)


def launch_counts(lazy_expr):
    """Graph-level launch counts (exact, backend-independent).

    Counts compute nodes only — ``InputNode``/``ConstantNode`` leaves are
    free (no kernel), so ``fusion_stats()``'s ``materialized`` (which walks
    every node) would overcount.
    """
    from awkward._connect.lazy import _fusion
    from awkward._connect.lazy._ir import ConstantNode, InputNode

    def launches(root):
        return sum(
            1
            for n in _fusion._walk(root)
            if not isinstance(n, (ConstantNode, InputNode))
        )

    root = lazy_expr.ir_node
    return launches(root), launches(_fusion.fuse(root))


def run(backend, sizes, depth, repeats):
    results = {"backend": backend, "depth": depth, "sizes": list(sizes), "data": {}}
    for n in sizes:
        arr = make_jagged(n, backend)
        content_bytes = int(arr.layout.content.data.nbytes)
        entry = {"content_bytes": content_bytes, "pipelines": {}}
        for name, (lazy_make, eager_make) in PIPELINES.items():
            mk_lazy = (lambda lm=lazy_make: lm(arr, depth))
            li, lf = launch_counts(mk_lazy())
            row = {
                "launches_interp": li,
                "launches_fused": lf,
                # traffic model: 1 read + 1 write of flat content per launch
                "traffic_interp_bytes": 2 * li * content_bytes,
                "traffic_fused_bytes": 2 * lf * content_bytes,
                "eager_s": time_best(
                    lambda em=eager_make: em(arr, depth), backend, repeats
                ),
                "interp_s": time_best(
                    lambda: mk_lazy().compute(fuse=False), backend, repeats
                ),
                "fused_s": time_best(
                    lambda: mk_lazy().compute(fuse=True), backend, repeats
                ),
            }
            for mode, fn in (
                ("eager", lambda em=eager_make: em(arr, depth)),
                ("interp", lambda: mk_lazy().compute(fuse=False)),
                ("fused", lambda: mk_lazy().compute(fuse=True)),
            ):
                alloc = measured_alloc_bytes(fn, backend)
                if alloc is not None:
                    row[f"alloc_{mode}_bytes"] = alloc
            entry["pipelines"][name] = row
            print(
                f"[{backend}] n={n:>10,} {name:<7} "
                f"eager {row['eager_s'] * 1e3:8.2f} ms  "
                f"interp {row['interp_s'] * 1e3:8.2f} ms  "
                f"fused {row['fused_s'] * 1e3:8.2f} ms  "
                f"launches {li}->{lf}"
            )
        results["data"][str(n)] = entry
    return results


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------


def _load(paths):
    by_backend = {}
    for p in paths:
        d = json.load(open(p))
        by_backend[d["backend"]] = d
    return by_backend


def _sizes_of(d):
    return sorted(int(k) for k in d["data"])


def plot_runtime(by_backend, pipeline="chain"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"cpu": "--", "cuda": "-"}
    colors = {"eager": SLOW, "interp": GREY, "fused": FAST}
    for backend, d in by_backend.items():
        sizes = _sizes_of(d)
        for mode in ("eager", "interp", "fused"):
            ys = [d["data"][str(n)]["pipelines"][pipeline][f"{mode}_s"] for n in sizes]
            ax.plot(
                sizes,
                ys,
                styles.get(backend, "-"),
                marker="o",
                color=colors[mode],
                label=f"{backend} {mode}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("events (outer lists)")
    ax.set_ylabel("runtime [s]")
    depth = next(iter(by_backend.values()))["depth"]
    ax.set_title(f"Lazy IR runtime vs dataset size ({pipeline}, depth {depth})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "lazy_ir_runtime_vs_size.png", dpi=160)
    plt.close(fig)


def plot_speedup(by_backend, pipeline="chain"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if "cpu" in by_backend and "cuda" in by_backend:
        cpu_d, gpu_d = by_backend["cpu"], by_backend["cuda"]
        sizes = sorted(set(_sizes_of(cpu_d)) & set(_sizes_of(gpu_d)))
        x = np.arange(len(sizes))
        w = 0.35
        for off, (mode, color) in zip(
            (-w / 2, w / 2), (("eager", SLOW), ("fused", FAST))
        ):
            sp = [
                cpu_d["data"][str(n)]["pipelines"][pipeline][f"{mode}_s"]
                / gpu_d["data"][str(n)]["pipelines"][pipeline][f"{mode}_s"]
                for n in sizes
            ]
            bars = ax.bar(x + off, sp, w, color=color, label=f"{mode}")
            ax.bar_label(bars, fmt="%.1fx", fontsize=8)
        ax.set_title(f"GPU speedup over CPU ({pipeline})")
        ax.set_ylabel("CPU time / GPU time")
    else:
        (backend, d), = by_backend.items()
        sizes = _sizes_of(d)
        x = np.arange(len(sizes))
        sp = [
            d["data"][str(n)]["pipelines"][pipeline]["eager_s"]
            / d["data"][str(n)]["pipelines"][pipeline]["fused_s"]
            for n in sizes
        ]
        bars = ax.bar(x, sp, 0.5, color=FAST, label="fused vs eager")
        ax.bar_label(bars, fmt="%.1fx", fontsize=8)
        ax.set_title(
            f"Fused-over-eager speedup on {backend} ({pipeline})\n"
            "(run both backends and re-plot for GPU-over-CPU)"
        )
        ax.set_ylabel("eager time / fused time")
    ax.set_xticks(x, [f"{n:,}" for n in sizes])
    ax.set_xlabel("events")
    ax.axhline(1.0, color=INK, lw=0.8, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "lazy_ir_gpu_speedup.png", dpi=160)
    plt.close(fig)


def plot_launches(by_backend):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = next(iter(by_backend.values()))  # backend-independent counts
    n0 = str(_sizes_of(d)[0])
    names = list(d["data"][n0]["pipelines"])
    x = np.arange(len(names))
    w = 0.35
    interp = [d["data"][n0]["pipelines"][p]["launches_interp"] for p in names]
    fused = [d["data"][n0]["pipelines"][p]["launches_fused"] for p in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - w / 2, interp, w, color=SLOW, label="eager / interpreter")
    b2 = ax.bar(x + w / 2, fused, w, color=FAST, label="fused")
    ax.bar_label(b1, fontsize=9)
    ax.bar_label(b2, fontsize=9)
    ax.set_xticks(x, [f"{p} (d={d['depth']})" if p == "chain" else p for p in names])
    ax.set_ylabel("kernel launches (graph-level)")
    ax.set_title("Lazy IR kernel launch count: fusion collapse")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "lazy_ir_kernel_launches.png", dpi=160)
    plt.close(fig)


def plot_traffic(by_backend, pipeline="chain"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # model series from any backend; measured series from cuda if present
    d = by_backend.get("cuda") or next(iter(by_backend.values()))
    sizes = _sizes_of(d)
    x = np.arange(len(sizes))
    w = 0.28
    gb = 1024**3
    model_i = [
        d["data"][str(n)]["pipelines"][pipeline]["traffic_interp_bytes"] / gb
        for n in sizes
    ]
    model_f = [
        d["data"][str(n)]["pipelines"][pipeline]["traffic_fused_bytes"] / gb
        for n in sizes
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w, model_i, w, color=SLOW, label="eager/interp (model)")
    ax.bar(x, model_f, w, color=FAST, label="fused (model)")
    row0 = d["data"][str(sizes[0])]["pipelines"][pipeline]
    if "alloc_interp_bytes" in row0:
        meas_i = [
            d["data"][str(n)]["pipelines"][pipeline]["alloc_interp_bytes"] / gb
            for n in sizes
        ]
        meas_f = [
            d["data"][str(n)]["pipelines"][pipeline]["alloc_fused_bytes"] / gb
            for n in sizes
        ]
        ax.bar(x + w, meas_i, w * 0.5, color=SLOW, alpha=0.45,
               label="interp (measured alloc, GPU)")
        ax.bar(x + w * 1.5, meas_f, w * 0.5, color=FAST, alpha=0.45,
               label="fused (measured alloc, GPU)")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{n:,}" for n in sizes])
    ax.set_xlabel("events")
    ax.set_ylabel("bytes moved / allocated [GiB]")
    ax.set_title(
        f"Lazy IR memory traffic ({pipeline}, depth {d['depth']})\n"
        "model: 1 read + 1 write of flat content per launch"
    )
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "lazy_ir_memory_traffic.png", dpi=160)
    plt.close(fig)


def make_plots(paths):
    FIGS.mkdir(exist_ok=True)
    by_backend = _load(paths)
    plot_runtime(by_backend)
    plot_speedup(by_backend)
    plot_launches(by_backend)
    plot_traffic(by_backend)
    for f in (
        "lazy_ir_runtime_vs_size.png",
        "lazy_ir_gpu_speedup.png",
        "lazy_ir_kernel_launches.png",
        "lazy_ir_memory_traffic.png",
    ):
        print("wrote", FIGS / f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--out", help="write results JSON")
    parser.add_argument("--plot", nargs="+", metavar="JSON",
                        help="plot from result JSON(s) instead of running")
    args = parser.parse_args()

    if args.plot:
        make_plots(args.plot)
        return

    results = run(args.backend, args.sizes, args.depth, args.repeats)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

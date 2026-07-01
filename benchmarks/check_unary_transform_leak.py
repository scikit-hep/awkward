"""Direct check for the cuda.compute `unary_transform` per-call leak (PR #4056).

Repeatedly runs a `unary_transform`-backed reducer (`ak.argmax(axis=1)`),
drains the awkward CUDA error/context queue after each call, and records the
CuPy memory-pool usage. If cuda.cccl is leak-free, used-bytes plateaus; if it
regresses, it grows ~linearly with iterations. (cuda-cccl >= 1.0.2 is required
and fixes the original leak; this is kept as a regression check.)

    python benchmarks/check_unary_transform_leak.py
    python benchmarks/check_unary_transform_leak.py --iters 500
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--n-lists", type=int, default=200_000)
    ap.add_argument("--avg-len", type=int, default=50)
    args = ap.parse_args()

    import cupy as cp
    import numpy as np

    import awkward as ak
    import awkward._connect.cuda as ak_cu

    # versions
    def _ver(*names):
        from importlib.metadata import PackageNotFoundError, version

        for n in names:
            try:
                return f"{n}={version(n)}"
            except PackageNotFoundError:
                continue
        return "cuda.cccl=?"

    print(
        "awkward",
        ak.__version__,
        "|",
        _ver("cuda-cccl", "cuda-compute", "cuda"),
        flush=True,
    )

    rng = np.random.default_rng(12345)
    counts = rng.integers(0, 2 * args.avg_len + 1, size=args.n_lists)
    offsets = np.zeros(args.n_lists + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    data = rng.standard_normal(int(counts.sum()))
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(cp.asarray(offsets)),
        ak.contents.NumpyArray(cp.asarray(data)),
    )
    arr = ak.Array(layout)

    pool = cp.get_default_memory_pool()

    def used_mb():
        return pool.used_bytes() / 1e6

    # warmup + drain
    ak.argmax(arr, axis=1)
    ak_cu.synchronize_cuda()
    base = used_mb()

    samples = []
    for i in range(args.iters):
        ak.argmax(arr, axis=1)
        ak_cu.synchronize_cuda()
        if i % max(1, args.iters // 10) == 0 or i == args.iters - 1:
            samples.append((i, used_mb()))

    print(f"\nbaseline after warmup: {base:.2f} MB")
    for i, mb in samples:
        print(f"  iter {i:5d}: {mb:9.2f} MB   (+{mb - base:8.2f})")

    first, last = samples[0][1], samples[-1][1]
    growth = (last - first) / max(1, samples[-1][0] - samples[0][0])
    print(f"\ngrowth ~ {growth * 1e3:.1f} KB/iter over {args.iters} iters")
    # ~output-size-per-iter or less => effectively flat (no leak)
    out_bytes_mb = (args.n_lists * 8) / 1e6
    verdict = "NO LEAK (flat)" if (last - first) < 2 * out_bytes_mb else "STILL LEAKING"
    print("verdict:", verdict)


if __name__ == "__main__":
    main()

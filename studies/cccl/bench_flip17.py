# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Before/after benchmark for the 17 kernels flipped from CuPy raw kernels
to cuda.compute (Phase 1 of studies/cccl/cuda-compute-migration-plan.md).

Times high-level operations that dispatch through the flipped kernels, at the
operation level rather than raw-kernel level so the harness works identically
on both sides of the flip (raw-kernel dispatch no longer exists afterwards).

Covered kernels -> operations:
  ListArray_getitem_next_at/range/array(+_advanced)  -> jagged slices on ListArray
  ListArray_getitem_jagged_expand                    -> jagged-boolean slice
  ListArray/ListOffsetArray_rpad_axis1               -> ak.pad_none(clip=False)
  ListOffsetArray_toRegularArray                     -> ak.to_regular
  ListOffsetArray_flatten_offsets                    -> ak.flatten (nested lists)
  ListOffsetArray_drop_none_indexes                  -> ak.drop_none
  ListArray_validity / IndexedArray_simplify         -> ak.validity_error / projections
  RegularArray_getitem_next_array_regularize         -> integer-array slice on regular
  UnionArray_flatten_length/_combine                 -> ak.flatten on a union
  UnionArray_nestedfill_tags_index                   -> ak.concatenate(union merge)

Also times initialize_cuda_kernels() (NVRTC blob compilation) — the flip
deletes 17 .cu files, so startup should improve.

Usage (on a GPU node, e.g. della):
    # on the commit BEFORE the flip:
    python studies/cccl/bench_flip17.py --out before.json
    # on the commit AFTER the flip:
    python studies/cccl/bench_flip17.py --out after.json
    # compare:
    python studies/cccl/bench_flip17.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def build_arrays(n_lists, avg):
    import cupy as cp

    import awkward as ak

    rng = np.random.default_rng(12345)
    counts = rng.integers(0, 2 * avg, n_lists)
    offsets = np.zeros(n_lists + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    content = rng.normal(0, 1, int(offsets[-1]))

    listoffset = ak.to_backend(
        ak.Array(
            ak.contents.ListOffsetArray(
                ak.index.Index64(offsets), ak.contents.NumpyArray(content)
            )
        ),
        "cuda",
    )
    # A genuine ListArray (starts/stops) so ListArray_* kernels dispatch
    listarray = ak.to_backend(
        ak.Array(
            ak.contents.ListArray(
                ak.index.Index64(offsets[:-1]),
                ak.index.Index64(offsets[1:]),
                ak.contents.NumpyArray(content),
            )
        ),
        "cuda",
    )
    regular = ak.to_backend(
        ak.Array(np.arange(n_lists * 4, dtype=np.float64).reshape(n_lists, 4)), "cuda"
    )
    nested = ak.to_backend(
        ak.Array(
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.arange(0, n_lists + 1, dtype=np.int64)),
                listoffset.layout,
            )
        ),
        "cuda",
    )
    with_none = ak.to_backend(
        ak.drop_none(  # build option-typed jagged: mask ~10% of elements
            ak.mask(listoffset, listoffset > -10)  # no-op mask keeps option type
        ),
        "cuda",
    )
    option_inner = ak.to_backend(
        ak.mask(listoffset, listoffset > 1.0),  # jagged with many None
        "cuda",
    )
    union = ak.to_backend(
        ak.concatenate(
            [
                ak.Array([[1.0, 2.0], [3.0]] * (n_lists // 2)),
                ak.Array([[1, 2, 3], [4]] * (n_lists // 2)),
            ],
            axis=0,
            mergebool=False,
        ),
        "cuda",
    )
    inner_mask = ak.to_backend(listoffset > 0.0, "cuda")
    idx = ak.to_backend(ak.Array(np.array([0, 1], dtype=np.int64)), "cuda")
    return {
        "listoffset": listoffset,
        "listarray": listarray,
        "regular": regular,
        "nested": nested,
        "with_none": with_none,
        "option_inner": option_inner,
        "union": union,
        "inner_mask": inner_mask,
        "idx": idx,
    }


def make_ops(a):
    import awkward as ak

    return {
        "getitem_next_at (arr[:, 0])": lambda: a["listarray"][:, 0],
        "getitem_next_range (arr[:, 1:3])": lambda: a["listarray"][:, 1:3],
        "getitem_next_array (arr[:, [0, 1]])": lambda: a["listarray"][:, a["idx"]],
        "jagged_expand (arr[jagged bool])": lambda: a["listarray"][a["inner_mask"]],
        "rpad_axis1 (ak.pad_none clip=False)": lambda: ak.pad_none(
            a["listarray"], 8, axis=1, clip=False
        ),
        "rpad_axis1_listoffset (ak.pad_none)": lambda: ak.pad_none(
            a["listoffset"], 8, axis=1, clip=False
        ),
        "toRegularArray (ak.to_regular)": lambda: ak.to_regular(
            a["regular"][:, :4], axis=1
        ),
        "flatten_offsets (ak.flatten nested)": lambda: ak.flatten(a["nested"], axis=1),
        "drop_none (ak.drop_none)": lambda: ak.drop_none(a["option_inner"], axis=1),
        "validity (ak.validity_error)": lambda: ak.validity_error(a["listarray"]),
        "regularize (regular[:, [0, 1]])": lambda: a["regular"][:, a["idx"]],
        "union_flatten (ak.flatten union)": lambda: ak.flatten(a["union"], axis=1),
        "union_concat (ak.concatenate)": lambda: ak.concatenate(
            [a["union"], a["union"]], axis=0
        ),
        "simplify (option projection)": lambda: ak.drop_none(a["with_none"]),
    }


def bench(fn, repeats):
    import cupy as cp

    stream = cp.cuda.get_current_stream()
    fn()  # warmup (JIT/adapter compilation excluded from timing)
    stream.synchronize()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        stream.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write results JSON")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--lists", type=int, default=200_000)
    parser.add_argument("--avg", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    if args.compare:
        before = json.load(open(args.compare[0]))
        after = json.load(open(args.compare[1]))
        print(f"{'operation':<42} {'before':>10} {'after':>10} {'ratio':>7}")
        for k in before:
            if k in after and isinstance(before[k], float):
                r = after[k] / before[k]
                flag = "OK" if r < 1.10 else ("~" if r < 1.35 else "SLOWER")
                print(
                    f"{k:<42} {before[k] * 1e3:>9.3f}ms {after[k] * 1e3:>9.3f}ms "
                    f"{r:>6.2f}x {flag}"
                )
        return

    # startup: NVRTC blob compilation time (fresh process required for a true
    # cold measurement; this captures it if run first)
    import cupy

    from awkward._connect import cuda as ak_cuda

    t0 = time.perf_counter()
    ak_cuda.initialize_cuda_kernels(cupy)
    startup = time.perf_counter() - t0

    arrays = build_arrays(args.lists, args.avg)
    results = {"initialize_cuda_kernels_s": startup}
    for name, fn in make_ops(arrays).items():
        try:
            results[name] = bench(fn, args.repeats)
            print(f"{name:<42} {results[name] * 1e3:>9.3f} ms")
        except Exception as exc:  # noqa: BLE001 — record and continue
            results[name] = f"ERROR: {exc}"
            print(f"{name:<42} ERROR: {exc}")
    print(f"{'initialize_cuda_kernels':<42} {startup * 1e3:>9.3f} ms")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

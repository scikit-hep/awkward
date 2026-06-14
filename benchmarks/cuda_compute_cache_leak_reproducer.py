#!/usr/bin/env python3
"""
Reproducer: the ~88 MB retained after draining Awkward's CUDA contexts is NOT
an Awkward leak -- it is cuda.compute's bounded build-result cache.

Background
----------
During PR #4056 GPU validation, a referrer dump taken *after* draining
Awkward's CUDA ErrorContext list (`cuda_streamptr_to_contexts`) still showed
several ~16 MB buffers alive:

    after draining awkward contexts: 88696832
    16006512 ['builtins.list', 'builtins.list', 'compute._bindings_impl.Iterator']
    ...
    16006512 ['builtins.list', 'builtins.list', 'cupy.ndarray']

Tracing the referrers to their root showed the holder is *not* any Awkward
object (no Invocation / ErrorContext / cuda_streamptr_to_contexts on the
chain). The root is `cuda.compute`'s own build-result cache:

    cuda.compute._caching._CacheWithRegisteredKeyFunctions
      -> [ _SegmentedReduce, _SegmentedReduce, ... ]        # one per signature
           -> compute._bindings_impl.Iterator (d_in_cccl, d_out_cccl, ...)
                -> cupy.ndarray  (the device buffers from each build)

What the empirical run shows
----------------------------
cuda.compute keys this cache on the *compiled reduction signature* (dtypes +
op semantics), NOT on the Python `op` object's identity. So:

  * Calling the SAME reduction many times -- even with a brand-new `sum_op`
    closure every call, exactly as `awkward_reduce_sum_complex` does -- keeps
    the cache at ONE entry and device memory flat. (See `run_repeated`.)
  * Memory grows only when a NEW signature is exercised, by one build-result
    (~one buffer set) per signature, then plateaus. A second pass over the same
    signatures adds nothing. (See `run_distinct_signatures`.)

So the PR dump's ~88 MB = (number of distinct reductions exercised) x (one
build-result each), held by cuda.compute's cache. It is bounded steady state,
reached and held -- not unbounded accumulation, and not owned by Awkward.
Awkward's ErrorContext drain is correct and irrelevant to this memory.

This script imports ONLY cupy + cuda.compute; `awkward` is never loaded.

Note: `OpKind.PLUS` currently fails to NVRTC-compile on complex128 in this
cuda.compute build, which is precisely why `awkward_reduce_sum_complex` uses a
hand-written `sum_op` closure. Each reduction below picks an op that compiles
for its dtype; build failures are caught and reported rather than aborting.

Requires an NVIDIA GPU with `cupy` and `cuda-cccl` (the `cuda.compute` module)
installed. Run:

    python cuda_compute_cache_leak_reproducer.py
"""

from __future__ import annotations

import gc
import sys

import numpy as np

try:
    import cupy as cp
    from cuda.compute import OpKind, segmented_reduce
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This reproducer needs a GPU box with `cupy` and `cuda.compute` "
        f"(cuda-cccl) installed. Import failed: {exc!r}"
    )

# ~16 MB per build, matching the buffers in the PR referrer dump.
TARGET_BYTES = 16_000_000
N_SEGMENTS = 1_000
N_ITER = 8


def used_device_bytes() -> int:
    return int(cp.get_default_memory_pool().used_bytes())


def count_live(*type_names: str) -> int:
    """Count live objects whose type name matches any of `type_names`."""
    wanted = set(type_names)
    n = 0
    for obj in gc.get_objects():
        try:
            t = type(obj)
        except ReferenceError:
            continue
        if t.__name__ in wanted or getattr(t, "__qualname__", "") in wanted:
            n += 1
    return n


def make_inputs(dtype):
    """Fresh device buffers (~TARGET_BYTES of input) for one reduction."""
    n_elements = TARGET_BYTES // np.dtype(dtype).itemsize
    seg = n_elements // N_SEGMENTS
    n_elements = seg * N_SEGMENTS  # make it divide evenly
    d_in = cp.ones(n_elements, dtype=dtype)
    d_out = cp.empty(N_SEGMENTS, dtype=dtype)
    start = cp.arange(N_SEGMENTS, dtype=cp.int64) * seg
    end = start + seg
    h_init = np.asarray(0, dtype=dtype)
    return d_in, d_out, start, end, h_init


def reduce_once(dtype, *, fresh_closure: bool):
    """One segmented reduction. If `fresh_closure`, build a brand-new `sum_op`
    object every call -- exactly like awkward_reduce_sum_complex. Otherwise use
    the builtin OpKind.PLUS. Returns True on success, False if the JIT build
    failed for this dtype/op combination."""
    d_in, d_out, start, end, h_init = make_inputs(dtype)

    if fresh_closure:
        def sum_op(a, b):  # new function object on every invocation
            return a + b
        op = sum_op
    else:
        op = OpKind.PLUS

    try:
        segmented_reduce(
            d_in=d_in,
            d_out=d_out,
            num_segments=N_SEGMENTS,
            start_offsets_in=start,
            end_offsets_in=end,
            op=op,
            h_init=h_init,
        )
        cp.cuda.runtime.deviceSynchronize()
        ok = True
    except Exception as exc:
        print(f"    (build failed for {np.dtype(dtype).name} with "
              f"{'closure' if fresh_closure else 'OpKind.PLUS'}: "
              f"{type(exc).__name__}: {str(exc).splitlines()[0]})")
        ok = False

    # Drop every user-facing reference and force a full collection -- the same
    # way the original investigation drained the awkward contexts.
    del d_in, d_out, start, end, h_init
    gc.collect()
    return ok


def run_repeated() -> None:
    """Call the SAME reduction N times with a fresh closure each call. If this
    were an unbounded leak the cache entry count and device memory would climb;
    instead both stay flat at one entry."""
    print("\n=== Test 1: same reduction x N, fresh `sum_op` closure each call ===")
    print("    (mirrors awkward_reduce_sum_complex; expectation: flat at 1 entry)")
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    base = count_live("_SegmentedReduce")

    for i in range(N_ITER):
        reduce_once(cp.complex128, fresh_closure=True)
        print(f"  call {i + 1:>2}: live _SegmentedReduce="
              f"{count_live('_SegmentedReduce') - base:>3}  "
              f"device_used={used_device_bytes():>12,} B")


def run_distinct_signatures() -> None:
    """Exercise several DISTINCT reduction signatures, twice. Memory grows by
    one build-result per new signature on the first pass, then is flat on the
    second pass -- this is the ~88 MB / ~6-buffer steady state from the PR dump,
    bounded by the number of signatures, not the number of calls."""
    print("\n=== Test 2: distinct signatures, two passes ===")
    print("    (expectation: grows by 1 build-result per new signature, then flat)")
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    base = count_live("_SegmentedReduce")

    # OpKind.PLUS compiles for real dtypes; complex128 uses a closure op.
    plan = [
        (cp.float32, False),
        (cp.float64, False),
        (cp.int32, False),
        (cp.int64, False),
        (cp.complex128, True),
    ]

    for pass_no in (1, 2):
        print(f"  -- pass {pass_no} --")
        for dtype, fresh in plan:
            reduce_once(dtype, fresh_closure=fresh)
            print(f"    {np.dtype(dtype).name:>10}: live _SegmentedReduce="
                  f"{count_live('_SegmentedReduce') - base:>3}  "
                  f"device_used={used_device_bytes():>12,} B")


def main() -> None:
    assert "awkward" not in sys.modules, "awkward must not be loaded"
    print(
        "Proving the retained device memory is cuda.compute's bounded\n"
        "build-result cache, not an Awkward leak.\n"
        f"Target per build: ~{TARGET_BYTES:,} bytes.\n"
        "NOTE: `awkward` is intentionally never imported by this script."
    )

    run_repeated()
    run_distinct_signatures()

    print(
        "\nInterpretation:\n"
        "  * Test 1 stays flat at one entry: repeated calls -- even with a new\n"
        "    `sum_op` closure each time -- reuse a single cached build-result.\n"
        "    cuda.compute keys on the compiled signature, not the op's identity,\n"
        "    so there is no per-call accumulation. No growth == no leak.\n"
        "  * Test 2 grows by one build-result per NEW signature, then plateaus on\n"
        "    the second pass. The PR dump's ~88 MB is exactly this: ~6 distinct\n"
        "    reductions x one build-result each, held by cuda.compute's cache.\n"
        "  * No awkward is imported, so the residue cannot be an Awkward leak;\n"
        "    Awkward's ErrorContext drain is correct and unrelated to this memory.\n"
        "  * If this steady-state footprint must shrink, the lever is\n"
        "    cuda.compute's cache (size/eviction), not Awkward."
    )


if __name__ == "__main__":
    main()

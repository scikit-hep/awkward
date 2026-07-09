# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Phase 1 probe: can cuda.compute compose multiple ops into ONE kernel?

Answers the gating question from ``cccl-followup-plan-revised.md`` (Open Q1):
*can cuda.compute compose/delay multiple ops into a single compiled kernel, or
must we build the DAG->single-kernel emitter ourselves?*

It contrasts three ways to run a depth-N element-wise chain (optionally reduced)
over one flat buffer, and times each with CUDA events across increasing depth:

  A. separate primitives  — N ``unary_transform`` calls, each writing a fresh
     intermediate buffer (the "no fusion" baseline).
  B. composed op          — ONE ``unary_transform`` whose op is the whole
     depth-N expression (fusion by folding the ops into a single custom op).
  C. iterator composition — nested ``TransformIterator`` (depth N) fed into ONE
     consuming primitive (``unary_transform`` for map, ``reduce_into`` for
     map+reduce). This is the idiom already shipping in ``_compute.py``
     (``_widen_for_reduce`` / ``_nonzero_for_reduce``).

Reading the result
------------------
* If B and C stay ~flat as depth grows while A scales ~linearly, then the whole
  chain runs as ONE kernel in B/C: cuda.compute composes multiple ops into one
  compiled kernel **when we drive it** via a composed op or composed iterators.
* A scaling linearly (and allocating an intermediate per step) shows cuda.compute
  does **not** auto-fuse a sequence of separate primitive *launches* — there is
  no deferred/lazy graph that fuses post-hoc. The fused kernel is emitted by the
  composition layer (our op / iterator), compiled once by cuda.compute.

For an exact kernel COUNT (not just timing), run under nsys:

    nsys profile -t cuda --stats=true python studies/cccl/probe_fusion_mechanism.py

Expected: A fires ~N transform kernels (+1 reduce); B fires 1; C fires 1.

Usage:  python studies/cccl/probe_fusion_mechanism.py [--n 2000000]
"""

from __future__ import annotations

import argparse
import functools

import numpy as np

try:
    import cupy as cp
    from cuda.compute import OpKind, TransformIterator, reduce_into, unary_transform
except ImportError as exc:  # pragma: no cover - needs a GPU
    raise SystemExit(f"this probe needs cupy + cuda.compute on a GPU: {exc}")

REPS = 100
DEPTHS = (1, 2, 4, 8, 16)
MUL, ADD = np.float64(1.001), np.float64(0.5)


@functools.cache
def _step():
    """One affine step ``x -> x*MUL + ADD`` (interned -> one compiled kernel)."""

    def f(x):
        return x * MUL + ADD

    return f


@functools.cache
def _chain_op(depth):
    """The whole depth-N chain folded into a single op (cache-stable)."""

    def f(x):
        y = x
        for _ in range(depth):
            y = y * MUL + ADD
        return y

    return f


def _cuda_time(fn, reps=REPS):
    fn()  # warmup: NVRTC / cuda.compute JIT + op-cache fill
    cp.cuda.Device().synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / reps


# ---- A. separate primitives: N transforms, N intermediates -----------------


def run_separate(x, depth):
    step = _step()
    buf = x
    for _ in range(depth):
        out = cp.empty_like(x)  # a fresh intermediate per op
        unary_transform(d_in=buf, d_out=out, op=step, num_items=x.size)
        buf = out
    return buf


# ---- B. composed op: one transform, one kernel -----------------------------


def run_composed_op(x, depth):
    out = cp.empty_like(x)
    unary_transform(d_in=x, d_out=out, op=_chain_op(depth), num_items=x.size)
    return out


# ---- C. iterator composition: nested TransformIterator, one consumer -------


def _nested_iter(x, depth):
    step = _step()
    it = x
    for _ in range(depth):
        it = TransformIterator(it, step)
    return it


def run_iter_transform(x, depth):
    out = cp.empty_like(x)
    unary_transform(d_in=_nested_iter(x, depth), d_out=out, op=_step(), num_items=x.size)
    return out


def run_iter_reduce(x, depth):
    # map^depth fused straight into a full reduction -> one kernel, no buffer
    out = cp.empty(1, dtype=x.dtype)
    reduce_into(
        d_in=_nested_iter(x, depth),
        d_out=out,
        op=OpKind.PLUS,
        num_items=x.size,
        h_init=np.asarray([0.0], dtype=x.dtype),
    )
    return out


def run_separate_reduce(x, depth):
    mapped = run_separate(x, depth)  # N transforms + N buffers
    out = cp.empty(1, dtype=x.dtype)
    reduce_into(
        d_in=mapped,
        d_out=out,
        op=OpKind.PLUS,
        num_items=x.size,
        h_init=np.asarray([0.0], dtype=x.dtype),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000, help="element count")
    args = ap.parse_args()

    x = cp.asarray(np.random.default_rng(0).random(args.n))
    print(f"element-wise chain over {args.n:,} float64 items, {REPS} reps\n")

    hdr = f"{'depth':>6}{'A separate ms':>15}{'B composed ms':>15}{'C iter ms':>12}{'A/B':>8}{'A/C':>8}"
    print("== map only (chain of transforms) ==")
    print(hdr)
    print("-" * len(hdr))
    for d in DEPTHS:
        a = _cuda_time(lambda d=d: run_separate(x, d))
        b = _cuda_time(lambda d=d: run_composed_op(x, d))
        c = _cuda_time(lambda d=d: run_iter_transform(x, d))
        print(f"{d:>6}{a:>15.3f}{b:>15.3f}{c:>12.3f}{a / b:>7.1f}x{a / c:>7.1f}x")

    hdr2 = f"{'depth':>6}{'A+reduce ms':>15}{'C iter+reduce ms':>18}{'speedup':>10}"
    print("\n== map + reduction (chain then sum) ==")
    print(hdr2)
    print("-" * len(hdr2))
    for d in DEPTHS:
        a = _cuda_time(lambda d=d: run_separate_reduce(x, d))
        c = _cuda_time(lambda d=d: run_iter_reduce(x, d))
        print(f"{d:>6}{a:>15.3f}{c:>18.3f}{a / c:>9.1f}x")

    # verdict from the depth-16 map-only row
    a16 = _cuda_time(lambda: run_separate(x, 16))
    b16 = _cuda_time(lambda: run_composed_op(x, 16))
    b1 = _cuda_time(lambda: run_composed_op(x, 1))
    flat = b16 / b1 < 4.0  # composed op barely grows 1 -> 16 ops
    scales = a16 / b16 > 4.0  # separate path much slower at depth 16
    print("\n---- verdict ----")
    print(f"composed-op time depth1->16: {b1:.3f} -> {b16:.3f} ms "
          f"({'flat -> ONE kernel' if flat else 'grows -> not one kernel'})")
    print(f"separate vs composed @depth16: {a16 / b16:.1f}x "
          f"({'separate launches N kernels' if scales else 'inconclusive'})")
    if flat and scales:
        print(
            "\nYES — cuda.compute composes multiple ops into ONE compiled kernel,\n"
            "but only when WE compose them (a folded custom op, or nested\n"
            "TransformIterators into a single consuming primitive). It does NOT\n"
            "auto-fuse a sequence of separate primitive launches -- there is no\n"
            "deferred graph. The fused kernel is emitted by the composition layer\n"
            "(our _fusion_codegen op / iterators) and compiled once by cuda.compute."
        )
    else:
        print("\nInconclusive on this GPU; inspect the tables and rerun under nsys.")


if __name__ == "__main__":
    main()

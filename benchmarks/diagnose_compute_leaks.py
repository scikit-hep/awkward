#!/usr/bin/env python3
"""
Pinpoint which `_compute.py` function leaks on a given CuPy-backed operation.

The leak signature established in PR #4056: a cuda.compute call
(`unary_transform` / `segmented_reduce` / `inclusive_scan` / `reduce_into`)
whose `op` is a Python closure that *captures a CuPy device array* as a free
variable. Such a closure differs every call, defeats cuda.compute's
build-result cache, and pins that call's output buffer forever.

This script monkeypatches those four entry points (on the `_compute` module, so
it intercepts the by-name imports the functions actually use), runs the target
operation once, and reports for every cuda.compute call:

    <_compute.py function that issued it>   op_captures_cupy_array=<bool>

Any row with ``op_captures_cupy_array=True`` is a leaker to vectorise. Edit
`target()` below to reproduce whatever op your failing memory test exercises.

Run on the GPU box:  python benchmarks/diagnose_compute_leaks.py
"""

from __future__ import annotations

import collections
import traceback

import numpy as np

try:
    import cupy as cp

    import awkward as ak
    import awkward._connect.cuda._compute as C
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f"needs a GPU box with cupy + awkward + cuda.compute: {exc!r}"
    ) from exc


def _op_captures_cupy_array(op) -> bool:
    cells = getattr(op, "__closure__", None) or ()
    for cell in cells:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, cp.ndarray):
            return True
        # closures sometimes capture a small list/tuple of arrays
        if isinstance(value, (list, tuple)) and any(
            isinstance(v, cp.ndarray) for v in value
        ):
            return True
    return False


def _calling_compute_function() -> str:
    for frame in reversed(traceback.extract_stack()):
        if frame.filename.endswith("_compute.py") and frame.name != "_wrap":
            return frame.name
    return "<unknown>"


# (caller, captures_array) -> count
_log: collections.Counter = collections.Counter()


def _install_probes():
    for entry in (
        "unary_transform",
        "segmented_reduce",
        "inclusive_scan",
        "reduce_into",
    ):
        original = getattr(C, entry)

        def make(original, entry):
            def _wrap(*args, **kwargs):
                op = kwargs.get("op")
                captures = _op_captures_cupy_array(op) if op is not None else False
                _log[(_calling_compute_function(), entry, captures)] += 1
                return original(*args, **kwargs)

            return _wrap

        setattr(C, entry, make(original, entry))


def target():
    """Reproduce the operation under test (test_missing_repeat_memory)."""
    rows, cols = 100_000, 16
    data = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    array = ak.to_backend(ak.Array(data), "cuda")
    slicer = [0, None, cols - 1]
    return lambda: array[:, slicer]


def main() -> None:
    _install_probes()
    fn = target()
    fn()  # one call is enough to see the whole code path
    cp.cuda.Device().synchronize()

    print(
        "cuda.compute calls during one operation "
        "(op_captures_cupy_array=True ==> leaks one buffer per call):\n"
    )
    if not _log:
        print("  (no cuda.compute calls intercepted)")
    leakers = []
    for (caller, entry, captures), n in sorted(_log.items()):
        flag = "LEAKS" if captures else "ok"
        print(f"  [{flag:5}] {caller:50} via {entry:18} x{n}")
        if captures:
            leakers.append(caller)
    if leakers:
        print("\nLeaking functions to vectorise:", ", ".join(sorted(set(leakers))))
    else:
        print(
            "\nNo array-capturing closures on this path -- the residue is "
            "elsewhere (e.g. a bounded cuda.compute cache, not a per-call leak)."
        )


if __name__ == "__main__":
    main()

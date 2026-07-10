# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""GPU regressions for the fusion layout/robustness fixes (needs a GPU).

These mirror the CPU regressions in ``tests/test_3792_lazy_fusion.py`` for the
CUDA path, covering the adversarial-review breaks that only reproduce on GPU:
silent-wrong results for strings / indexed layouts, hard crashes on
ListArray / RegularArray / non-finite constants, and mixed-backend divergence.
"""

from __future__ import annotations

import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


def _fell_back(expr):
    expr.compute(fuse=True)
    h = expr.executor.fused_hits
    return h["cuda"] == 0 and h["eager"] >= 1


def test_string_compare_falls_back_and_matches_eager():
    s = ak.Array(["ab", "cd"], backend="cuda")
    s2 = ak.Array(["ab", "ce"], backend="cuda")
    fused = (ak.cuda.lazy(s) == ak.cuda.lazy(s2)).compute(fuse=True)
    assert ak.to_list(fused) == ak.to_list(s == s2)  # per-string, not per-char


def test_listarray_does_not_crash_and_matches_eager():
    base = ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], backend="cuda")
    taken = base[[2, 0, 1]]  # ListArray
    expr = ak.cuda.lazy(taken) * 2
    assert _fell_back(expr)
    assert ak.to_list(expr.compute(fuse=True)) == ak.to_list(taken * 2)


def test_regular_array_does_not_crash_and_preserves_type():
    r = ak.to_regular(ak.Array([[1.0, 2.0], [3.0, 4.0]], backend="cuda"))
    expr = ak.cuda.lazy(r) * 2
    assert _fell_back(expr)
    assert str(ak.type(expr.compute(fuse=True))) == str(ak.type(r * 2))


def test_non_finite_constant_still_fuses():
    arr = ak.Array([[1.0, 2.0], [3.0]], backend="cuda")
    expr = ak.cuda.lazy(arr) * float("inf")
    out = expr.compute(fuse=True)
    assert expr.executor.fused_hits["cuda"] >= 1  # fused, not fallback
    assert ak.to_list(out) == ak.to_list(arr * float("inf"))


def test_mixed_backend_matches_eager_validity():
    cuda_la = ak.cuda.lazy(ak.Array([[1.0, 2.0]], backend="cuda"))
    cpu_la = ak.cpu.lazy(ak.Array([[1.0, 2.0]]))
    # fuse=False raises like eager; fuse=True must not silently succeed by
    # copying one operand across devices.
    with pytest.raises(Exception):  # noqa: B017
        (cuda_la + cpu_la).compute(fuse=False)
    with pytest.raises(Exception):  # noqa: B017
        (cuda_la + cpu_la).compute(fuse=True)


def test_deep_chain_falls_back_not_crash():
    arr = ak.Array([[1.0, 2.0], [3.0]], backend="cuda")
    expr = ak.cuda.lazy(arr)
    for _ in range(300):
        expr = expr + 1.0
    assert ak.to_list(expr.compute(fuse=True)) == ak.to_list(expr.compute(fuse=False))


def test_integer_powers_probe_does_not_break_valid_data():
    # _infer_out_dtype's np.ones(1) probe must not reject valid data whose real
    # exponents are all >= 0 (int ** negative in the probe only).
    a = ak.Array([[2, 3], [4]], backend="cuda")
    b = ak.Array([[3, 2], [2]], backend="cuda")
    out = (ak.cuda.lazy(a) ** ak.cuda.lazy(b)).compute(fuse=True)
    assert ak.to_list(out) == ak.to_list(a**b)


# ---- public ak.cuda.to_cccl_iterator: no silent-wrong / OOB on bad layouts --


def test_to_cccl_iterator_rejects_indexed_of_list():
    base = ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], backend="cuda")
    idx = ak.Array(
        ak.contents.IndexedArray(
            ak.index.Index64(cp.asarray([2, 0, 1], dtype=cp.int64)), base.layout
        )
    )
    with pytest.raises(NotImplementedError, match="to_packed"):
        ak.cuda.to_cccl_iterator(idx)


def test_to_cccl_iterator_normalizes_listarray():
    base = ak.Array([[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]], backend="cuda")
    taken = base[[2, 0, 1]]  # ListArray
    _it, meta = ak.cuda.to_cccl_iterator(taken)
    # offsets now exist (normalized to ListOffsetArray), count is the taken total
    assert meta["offsets"] is not None
    assert meta["count"] == 6

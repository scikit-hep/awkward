# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

"""Regressions from the adversarial review of the lazy IR (needs a GPU).

Every test pins a CUDA-path case that either crashed where eager worked
(ListArray KeyError, RegularArray NotImplementedError, non-finite constants,
dtype-probe failures) or returned silently wrong device results
(IndexedArray-of-lists permutation, IndexedOptionArray out-of-bounds reads,
per-character string comparison, silent cpu->cuda migration of mixed-backend
expressions).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import awkward as ak

cp = pytest.importorskip("cupy")


@pytest.fixture
def arr():
    return ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]], backend="cuda")


@pytest.fixture
def arr_cpu():
    return ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]])


def _as_list(array):
    return ak.to_list(ak.to_backend(array, "cpu"))


# ----------------------------------------------------------------------
# Layouts that crashed or silently permuted wrong data
# ----------------------------------------------------------------------


def test_listarray_layout_matches_eager(arr, arr_cpu):
    carried = arr[[2, 0, 1]]  # ListArray (starts/stops): buffers have no -offsets
    expr = ak.cuda.lazy(carried) * 2
    assert _as_list(expr.compute(fuse=True)) == ak.to_list(arr_cpu[[2, 0, 1]] * 2)


def test_listarray_public_iterator(arr):
    carried = arr[[2, 0, 1]]
    it, meta = ak.cuda.to_cccl_iterator(carried)
    assert meta["count"] == 9
    assert cp.asarray(meta["offsets"]).tolist() == [0, 4, 7, 9]
    assert cp.asarray(it).tolist() == pytest.approx(
        ak.to_list(ak.flatten(ak.to_backend(carried, "cpu")))
    )


def test_indexed_array_of_lists_matches_eager(arr, arr_cpu):
    lay = ak.contents.IndexedArray(
        ak.index.Index64(cp.asarray(np.array([2, 0, 1]))), arr.layout
    )
    indexed = ak.Array(lay)
    expected = ak.to_list(ak.to_backend(indexed, "cpu") * 2)
    expr = ak.cuda.lazy(indexed) * 2
    assert _as_list(expr.compute(fuse=True)) == expected
    assert _as_list(expr.compute(fuse=False)) == expected


def test_indexed_array_of_lists_public_iterator(arr):
    # The list-level index must be applied to lists (via projection), never to
    # the flattened elements.
    lay = ak.contents.IndexedArray(
        ak.index.Index64(cp.asarray(np.array([2, 0, 1]))), arr.layout
    )
    indexed = ak.Array(lay)
    it, meta = ak.cuda.to_cccl_iterator(indexed)
    assert meta["count"] == 9
    assert cp.asarray(meta["offsets"]).tolist() == [0, 4, 7, 9]
    assert cp.asarray(it).tolist() == pytest.approx(
        ak.to_list(ak.flatten(ak.to_backend(indexed, "cpu")))
    )


def test_indexed_option_array_iterator_raises():
    # -1 (None) indices have no iterator representation; an out-of-bounds
    # device read is never acceptable.
    a = ak.Array([[1.0, 2.0], None, [3.0]], backend="cuda")
    with pytest.raises(NotImplementedError, match="missing values"):
        ak.cuda.to_cccl_iterator(a)


def test_option_lists_lazy_compute_matches_eager():
    a = ak.Array([[1.0, 2.0], None, [3.0]], backend="cuda")
    expected = ak.to_list(ak.to_backend(a, "cpu") * 2)
    expr = ak.cuda.lazy(a) * 2
    assert _as_list(expr.compute(fuse=True)) == expected


def test_regular_array_matches_eager_and_preserves_type():
    r = ak.to_regular(ak.Array([[1.0, 2.0], [3.0, 4.0]], backend="cuda"), axis=1)
    expr = ak.cuda.lazy(r) * 2
    fused = expr.compute(fuse=True)
    eager = r * 2
    assert str(fused.type) == str(eager.type)
    assert _as_list(fused) == _as_list(eager)


def test_flat_string_equality_matches_eager():
    s = ak.Array(["ab", "cd"], backend="cuda")
    s2 = ak.Array(["ab", "ce"], backend="cuda")
    expected = ak.to_list(ak.Array(["ab", "cd"]) == ak.Array(["ab", "ce"]))
    expr = ak.cuda.lazy(s) == ak.cuda.lazy(s2)
    assert _as_list(expr.compute(fuse=True)) == expected == [True, False]
    assert expr.executor.fused_hits["cuda"] == 0  # declined, not silently wrong


# ----------------------------------------------------------------------
# Codegen crashes on valid data (must fall back, not raise)
# ----------------------------------------------------------------------


def test_inf_constant_matches_eager(arr, arr_cpu):
    expr = ak.cuda.lazy(arr) * float("inf")
    assert _as_list(expr.compute(fuse=True)) == ak.to_list(arr_cpu * float("inf"))


def test_int_pow_negative_probe_does_not_crash():
    # The dtype probe evaluates the op on ones: 1 ** (1 - 2) raises for ints
    # even though the real exponents (b - 2 >= 0) are all valid.
    ia = ak.Array([[2, 3], [4]], backend="cuda")
    ib = ak.Array([[2, 2], [2]], backend="cuda")
    expected = ak.to_list(ak.Array([[2, 3], [4]]) ** (ak.Array([[2, 2], [2]]) - 2))
    expr = ak.cuda.lazy(ia) ** (ak.cuda.lazy(ib) - 2)
    assert _as_list(expr.compute(fuse=True)) == expected


def test_division_probe_emits_no_warning():
    # 1 / (1 - 1) in the dtype probe must not warn (errors under pytest's
    # warnings-as-errors) when the real data has no zeros.
    ia = ak.Array([[4.0, 6.0], [8.0]], backend="cuda")
    ib = ak.Array([[3.0, 4.0], [5.0]], backend="cuda")
    expected = ak.to_list(
        ak.Array([[4.0, 6.0], [8.0]]) / (ak.Array([[3.0, 4.0], [5.0]]) - 1)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expr = ak.cuda.lazy(ia) / (ak.cuda.lazy(ib) - 1)
        assert _as_list(expr.compute(fuse=True)) == expected


def test_deep_chain_on_cuda(arr, arr_cpu):
    n = 300
    expr = ak.cuda.lazy(arr)
    for _ in range(n):
        expr = expr + 1
    assert _as_list(expr.compute(fuse=True)) == ak.to_list(arr_cpu + n)


# ----------------------------------------------------------------------
# Mixed backends: fuse=True must fail exactly like eager, not migrate data
# ----------------------------------------------------------------------


def test_mixed_backends_raise_like_eager(arr, arr_cpu):
    with pytest.raises(ValueError):
        arr + arr_cpu  # eager refuses mixed backends
    expr = ak.cuda.lazy(arr) + ak.cpu.lazy(arr_cpu)
    with pytest.raises(ValueError):
        expr.compute(fuse=True)


# ----------------------------------------------------------------------
# The fixes must not have knocked out the real fused kernels
# ----------------------------------------------------------------------


def test_elementwise_still_takes_cuda_kernel(arr, arr_cpu):
    expr = ak.cuda.lazy(arr) * 2 + 1
    assert _as_list(expr.compute(fuse=True)) == ak.to_list(arr_cpu * 2 + 1)
    assert expr.executor.fused_hits["cuda"] == 1


def test_fused_sum_still_takes_cuda_kernel(arr, arr_cpu):
    expr = (ak.cuda.lazy(arr) * 2).sum()
    assert _as_list(expr.compute(fuse=True)) == ak.to_list(ak.sum(arr_cpu * 2, axis=-1))
    assert expr.executor.fused_hits["cuda"] == 1


def test_listarray_leaves_still_fuse_on_device(arr, arr_cpu):
    # ListArray columns are normalized (to_ListOffsetArray64), not rejected.
    carried = arr[[2, 0, 1]]
    expr = ak.cuda.lazy(carried) * 2 + 1
    assert _as_list(expr.compute(fuse=True)) == ak.to_list(arr_cpu[[2, 0, 1]] * 2 + 1)
    assert expr.executor.fused_hits["cuda"] == 1


def test_transform_lists_ragged_raises():
    from awkward._connect.cuda.helpers import transform_lists

    ragged = ak.Array([[1.0, 2.0, 3.0], [4.0, 5.0], [6.0]], backend="cuda")
    out = ak.Array([[0.0], [0.0], [0.0]], backend="cuda")
    with pytest.raises(ValueError, match="exactly 2"):
        transform_lists(ragged, out, 2, lambda x, y: x + y)

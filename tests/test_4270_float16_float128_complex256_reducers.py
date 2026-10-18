# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import itertools

import numpy as np
import pytest

import awkward as ak

# float16, float128 and complex256 have no compiled reducer/sort kernels. They
# use float32 intermediates for float16 reductions and original-precision NumPy
# operations or exact rank keys for extended-precision reductions. Sort gathers
# original values through an argsort permutation, correcting float128 collisions
# in float64 keys. Complex arrays (any width) give a clear TypeError from
# sort/argsort, which awkward does not support.
#
# float128/complex256 are platform-dependent (e.g. macOS arm64 has neither); the
# tests for them skip where NumPy doesn't provide them.


ROWS = [[1.0, 2.0, 3.0], [4.0, 5.0]]  # exactly representable in float16


def _jagged(dtype):
    return ak.values_astype(ak.Array(ROWS), dtype)


@pytest.mark.parametrize(
    ("op", "npop"),
    [
        (ak.sum, np.sum),
        (ak.prod, np.prod),
        (ak.min, np.min),
        (ak.max, np.max),
        (ak.any, np.any),
        (ak.all, np.all),
        (ak.count_nonzero, np.count_nonzero),
        (ak.argmin, np.argmin),
        (ak.argmax, np.argmax),
    ],
)
def test_float16_reducers_axis1(op, npop):
    arr = _jagged(np.float16)
    got = ak.to_list(op(arr, axis=1))
    exp = [npop(np.array(r, dtype=np.float16)) for r in ROWS]
    assert got == pytest.approx([float(e) for e in exp])


@pytest.mark.parametrize(
    ("op", "npop"),
    [(ak.sum, np.sum), (ak.prod, np.prod), (ak.min, np.min), (ak.max, np.max)],
)
def test_float16_reducers_axis_none(op, npop):
    arr = _jagged(np.float16)
    flat = np.array([x for r in ROWS for x in r], dtype=np.float16)
    assert op(arr, axis=None) == pytest.approx(float(npop(flat)))


def test_float16_value_reducers_preserve_dtype():
    arr = _jagged(np.float16)
    for op in (ak.sum, ak.prod, ak.min, ak.max, ak.sort):
        assert "float16" in str(op(arr, axis=1).type)


def test_float16_sort_argsort():
    arr = _jagged(np.float16)
    assert ak.to_list(ak.sort(arr, axis=1)) == [[1.0, 2.0, 3.0], [4.0, 5.0]]
    assert ak.to_list(ak.sort(arr, axis=1, ascending=False)) == [
        [3.0, 2.0, 1.0],
        [5.0, 4.0],
    ]
    assert ak.to_list(ak.argsort(arr, axis=1)) == [[0, 1, 2], [0, 1]]
    assert "int64" in str(ak.argsort(arr, axis=1).type)


def test_float16_mean_std_var():
    # Check derived statistics as well as the primitive reducers, using a
    # float64 NumPy reference and tolerances appropriate for float16 results.
    arr = _jagged(np.float16)
    flat = np.array([x for r in ROWS for x in r], dtype=np.float64)
    assert ak.mean(arr, axis=None) == pytest.approx(np.mean(flat), rel=1e-2)
    assert ak.std(arr, axis=None) == pytest.approx(np.std(flat), rel=1e-2)
    assert ak.var(arr, axis=None) == pytest.approx(np.var(flat), rel=1e-2)


def test_float16_flat_sort_from_issue():
    # The exact reproduce from the issue.
    out = ak.sort(ak.Array(np.array([2.0, 1.0], dtype=np.float16)))
    assert ak.to_list(out) == [1.0, 2.0]
    assert "float16" in str(out.type)


@pytest.mark.parametrize("width", ["complex64", "complex128"])
def test_complex_sort_argsort_raise_typeerror(width):
    arr = ak.values_astype(ak.Array([[1 + 1j, 2 + 0j], [3 - 1j]]), getattr(np, width))
    with pytest.raises(TypeError, match="not supported"):
        ak.sort(arr, axis=1)
    with pytest.raises(TypeError, match="not supported"):
        ak.argsort(arr, axis=1)


def test_float16_nan_inf_sort_and_reducers():
    # awkward's (arg)sort is NaN-aware and pushes NaN to the low end (ascending),
    # which differs from NumPy. The carry-based sort gathers the original float16
    # values, so the NaN and both infinities survive intact.
    data = np.array([2.0, np.nan, 1.0, np.inf, -np.inf], dtype=np.float16)
    out = ak.sort(ak.Array(data))
    vals = np.asarray(out.layout.data)
    assert np.isnan(vals[0])
    assert vals[1] == -np.inf
    assert vals[-1] == np.inf
    assert "float16" in str(out.type)
    # reducers propagate inf like NumPy's float32 accumulation
    assert ak.max(ak.Array(np.array([1.0, 2.0, np.inf], dtype=np.float16))) == np.inf
    assert ak.min(ak.Array(np.array([1.0, 2.0, -np.inf], dtype=np.float16))) == -np.inf
    assert np.isinf(float(ak.sum(ak.Array(np.array([np.inf, 1.0], dtype=np.float16)))))


# --- extended precision (platform-gated) ------------------------------------


@pytest.mark.skipif(not hasattr(np, "float128"), reason="no float128 on this platform")
@pytest.mark.parametrize(
    ("op", "npop"),
    [
        (ak.sum, np.sum),
        (ak.prod, np.prod),
        (ak.min, np.min),
        (ak.max, np.max),
        (ak.all, np.all),
        (ak.argmin, np.argmin),
        (ak.count_nonzero, np.count_nonzero),
    ],
)
def test_float128_reducers_axis1(op, npop):
    arr = _jagged(np.float128)
    got = ak.to_list(op(arr, axis=1))
    exp = [npop(np.array(r, dtype=np.float128)) for r in ROWS]
    assert got == pytest.approx([float(e) for e in exp])


@pytest.mark.skipif(not hasattr(np, "float128"), reason="no float128 on this platform")
def test_float128_sort_preserves_dtype():
    arr = _jagged(np.float128)
    assert ak.to_list(ak.sort(arr, axis=1)) == [[1.0, 2.0, 3.0], [4.0, 5.0]]
    assert "float128" in str(ak.sort(arr, axis=1).type)


@pytest.mark.skipif(not hasattr(np, "float128"), reason="no float128 on this platform")
def test_float128_sort_preserves_exact_values():
    # `a` differs from 1.0 only at float128 precision: 2**-60 is below float64's
    # ~2**-52 resolution, so casting `a` to float64 rounds it to exactly 1.0. A
    # cast-back sort would return two 1.0s; the carry-based sort must return the
    # original multiset with both values still distinct.
    one = np.float128(1)
    a = one + np.float128(2) ** -60
    assert a != one  # distinct at float128 precision
    assert np.float64(a) == np.float64(one)  # indistinguishable in float64

    out = ak.sort(ak.Array(np.array([a, one], dtype=np.float128)))
    vals = np.asarray(out.layout.data)
    # Sorted in float128, not by the float64 keys (which are equal).
    assert vals[0] == one
    assert vals[1] == a
    assert vals[1] != one  # the extended-precision value was not rounded away
    assert "float128" in str(out.type)


@pytest.mark.skipif(not hasattr(np, "float128"), reason="no float128 on this platform")
def test_float128_argsort_orders_by_float128():
    # Two values equal after the float64 cast but distinct at float128.
    one = np.float128(1)
    a = one + np.float128(2) ** -60  # a > one, but == one in float64
    arr = ak.Array(np.array([a, one], dtype=np.float128))
    assert ak.to_list(ak.argsort(arr)) == [1, 0]
    assert ak.to_list(ak.argsort(arr, ascending=False)) == [0, 1]


@pytest.mark.skipif(not hasattr(np, "float128"), reason="no float128 on this platform")
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
def test_float128_sort_collisions_overflow_underflow(ascending, stable):
    # Values that collide in float64: sub-ulp differences, overflow to inf,
    # underflow to 0; plus NaN and signed zero, which keep the kernel semantics.
    one = np.float128(1)
    e = np.float128(2) ** -60
    data = np.array(
        [one + 2 * e, np.nan, one, -np.float128(0), one + e,
         np.float128("1e500"), np.float128("1e400"), np.float128("1e-400"), 0, one + e],
        dtype=np.float128,
    )  # fmt: skip
    arr = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 4, 10])), ak.contents.NumpyArray(data)
        )
    )
    out = ak.sort(arr, axis=1, ascending=ascending, stable=stable)
    index = ak.argsort(arr, axis=1, ascending=ascending, stable=stable)
    assert ak.to_list(out) == ak.to_list(arr[index]) or np.isnan(out[0, 0])
    for row in ak.to_list(out):
        finite = [x for x in row if not np.isnan(x)]
        pairs = itertools.pairwise(finite)
        assert all((x <= y) if ascending else (x >= y) for x, y in pairs)
    expected = [4, 3, 0, 5, 2, 1] if ascending else [1, 2, 0, 5, 3, 4]
    assert ak.to_list(index[1]) == expected


@pytest.mark.skipif(not hasattr(np, "float128"), reason="no float128 on this platform")
def test_float128_reducers_exact():
    one = np.float128(1)
    a = one + np.float128(2) ** -60
    tiny = np.float128("1e-4000")
    data = np.array(
        [one, a, np.float128("1e400"), np.float128("1e500"), tiny, tiny, 1e16, 1, 1],
        dtype=np.float128,
    )
    arr = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 2, 4, 6, 9])), ak.contents.NumpyArray(data)
        )
    )
    assert ak.max(arr, axis=1)[0] == a
    assert ak.argmax(arr, axis=1)[0] == 1
    assert ak.argmin(ak.Array(np.array([a, one], dtype=np.float128))) == 1
    assert ak.max(ak.Array(np.array([one, a], dtype=np.float128))) == a
    assert ak.max(arr, axis=1)[1] == np.float128("1e500")
    assert ak.sum(arr, axis=1)[1] == np.float128("1e400") + np.float128("1e500")
    assert ak.sum(arr, axis=1)[3] - np.float128(1e16) == 2
    assert ak.min(arr, axis=1)[2] == tiny
    assert ak.to_list(ak.count_nonzero(arr, axis=1)) == [2, 2, 2, 3]
    assert ak.to_list(ak.any(arr, axis=1)) == [True, True, True, True]
    assert ak.min(arr[:1], axis=1, initial=1.0)[0] == one
    assert ak.max(arr[:1], axis=1, initial=1.0)[0] == a


@pytest.mark.skipif(not hasattr(np, "float128"), reason="no float128 on this platform")
def test_float128_unique_keeps_distinct_values():
    one = np.float128(1)
    a = one + np.float128(2) ** -60
    arr = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 3, 3, 5])),
            ak.contents.NumpyArray(np.array([a, one, a, one, one], dtype=np.float128)),
        )
    )
    out = ak._do.unique(arr.layout, axis=-1)
    assert ak.to_list(out) == [[one, a], [], [one]]
    assert ak._do.unique(arr.layout.content, axis=None).length == 2


def test_float16_sort_record_field_longer_than_record():
    # A record's field may be longer than the record: positions past offsets[-1]
    # stay in place, as in the float32 path.
    for dtype in (np.float16, np.float32):
        content = ak.contents.NumpyArray(np.array([3.0, 1.0, 2.0], dtype))
        rec = ak.Array(ak.contents.RecordArray([content], ["x"], length=2))
        assert ak.to_list(ak.sort(rec, axis=0)) == [{"x": 1.0}, {"x": 3.0}]


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="no complex256 on this platform"
)
@pytest.mark.parametrize(
    ("op", "npop"),
    [
        (ak.sum, np.sum),
        (ak.prod, np.prod),
        (ak.count_nonzero, np.count_nonzero),
        (ak.min, np.min),
        (ak.max, np.max),
        (ak.argmin, np.argmin),
        (ak.argmax, np.argmax),
    ],
)
def test_complex256_reducers_axis1(op, npop):
    arr = ak.values_astype(ak.Array(ROWS), np.complex256)
    got = ak.to_list(op(arr, axis=1))
    exp = [npop(np.array(r, dtype=np.complex256)) for r in ROWS]
    assert got == pytest.approx([complex(e) for e in exp])


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="no complex256 on this platform"
)
def test_complex256_all_any():
    # all/any test the original complex256 values against zero and return bool.
    arr = ak.values_astype(
        ak.Array([[1 + 0j, 2 + 0j], [0 + 0j, 0 + 0j]]), np.complex256
    )
    assert ak.to_list(ak.all(arr, axis=1)) == [True, False]
    assert ak.to_list(ak.any(arr, axis=1)) == [True, False]


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="no complex256 on this platform"
)
def test_complex256_sort_raises_like_complex128():
    arr = ak.values_astype(ak.Array([[1 + 1j, 2 + 0j], [3 - 1j]]), np.complex256)
    with pytest.raises(TypeError, match="not supported"):
        ak.sort(arr, axis=1)


def _categorical_float(dtype):
    # A categorical array keeps its unique values in the content and indexes into
    # them; ak.is_valid / ak.validity_error check that the content is unique,
    # which dispatches _is_unique -> _unique -> awkward_sort by the value dtype.
    # Before this fix that raised KeyError for float16/float128 content.
    content = ak.contents.NumpyArray(np.array([1.0, 2.0, 3.0], dtype=dtype))
    index = ak.index.Index64(np.array([0, 1, 1, 2, 0], dtype=np.int64))
    return ak.Array(
        ak.contents.IndexedArray(
            index, content, parameters={"__array__": "categorical"}
        )
    )


def test_float16_categorical_is_valid():
    arr = _categorical_float(np.float16)
    assert ak.to_list(arr) == [1.0, 2.0, 2.0, 3.0, 1.0]
    assert ak.is_valid(arr)
    assert ak.validity_error(arr) == ""


@pytest.mark.skipif(not hasattr(np, "float128"), reason="no float128 on this platform")
def test_float128_categorical_is_valid():
    # Categorical validation checks uniqueness in the original float128 dtype.
    arr = _categorical_float(np.float128)
    assert ak.is_valid(arr)
    assert ak.validity_error(arr) == ""


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        pytest.param(
            getattr(np, "float128", None),
            marks=pytest.mark.skipif(
                not hasattr(np, "float128"), reason="no float128 on this platform"
            ),
        ),
    ],
)
def test_unique_per_list_casts_back_through_list_nodes(dtype):
    # With an axis (negaxis is not None), _unique returns a ListOffsetArray
    # wrapping the unique values rather than a bare NumpyArray. For float16 the
    # cast-back walk restores only the leaf dtype; float128 computes unique
    # values directly in its original precision. Both preserve the list node.
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 3, 3, 6], dtype=np.int64)),
        ak.contents.NumpyArray(np.array([2.0, 1.0, 2.0, 3.0, 3.0, 0.5], dtype=dtype)),
    )
    out = ak._do.unique(layout, axis=-1)
    assert isinstance(out, ak.contents.ListOffsetArray)
    assert out.content.dtype == np.dtype(dtype)
    assert ak.to_list(out) == [[1.0, 2.0], [], [0.5, 3.0]]


def test_unique_flat_float16_keeps_dtype():
    layout = ak.contents.NumpyArray(np.array([3.0, 1.0, 3.0, 2.0], dtype=np.float16))
    out = ak._do.unique(layout, axis=None)
    assert isinstance(out, ak.contents.NumpyArray)
    assert out.dtype == np.dtype(np.float16)
    assert ak.to_list(out) == [1.0, 2.0, 3.0]

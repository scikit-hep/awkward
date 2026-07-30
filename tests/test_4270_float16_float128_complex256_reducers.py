# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import numpy as np
import pytest

import awkward as ak

# float16, float128 and complex256 have no compiled reducer/sort kernels. They
# are now reduced/sorted through the nearest supported dtype (float16 -> float32,
# float128 -> float64, complex256 -> complex128) and value-preserving results are
# cast back, so every reducer and ak.sort/ak.argsort work instead of raising
# KeyError. Complex arrays (any width) give a clear TypeError from sort/argsort,
# since complex numbers have no total order.
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
    # These build on the float64-accumulator sum (awkward_reduce_sum_float64_*),
    # reached via the float16 -> float32 cast.
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
    with pytest.raises(TypeError, match="total order"):
        ak.sort(arr, axis=1)
    with pytest.raises(TypeError, match="total order"):
        ak.argsort(arr, axis=1)


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


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="no complex256 on this platform"
)
@pytest.mark.parametrize(
    ("op", "npop"),
    [(ak.sum, np.sum), (ak.prod, np.prod), (ak.count_nonzero, np.count_nonzero)],
)
def test_complex256_reducers_axis1(op, npop):
    arr = ak.values_astype(ak.Array(ROWS), np.complex256)
    got = ak.to_list(op(arr, axis=1))
    exp = [npop(np.array(r, dtype=np.complex256)) for r in ROWS]
    assert got == pytest.approx([complex(e) for e in exp])


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="no complex256 on this platform"
)
def test_complex256_sort_raises_like_complex128():
    arr = ak.values_astype(ak.Array([[1 + 1j, 2 + 0j], [3 - 1j]]), np.complex256)
    with pytest.raises(TypeError, match="total order"):
        ak.sort(arr, axis=1)

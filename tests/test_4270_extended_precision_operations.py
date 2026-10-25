# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble, np.clongdouble])
@pytest.mark.parametrize("axis", [None, -1, 0])
@pytest.mark.parametrize(
    "name",
    ["sum", "prod", "min", "max", "argmin", "argmax", "any", "all", "count_nonzero"],
)
def test_reductions(dtype, axis, name):
    data = np.array([[3, 1, 2], [2, 4, 1]], dtype=dtype)
    if data.dtype.kind == "c":
        data.imag = [[1, 2, 3], [2, 1, 0]]
    array = ak.Array(data)
    result = getattr(ak, name)(array, axis=axis)
    expected = getattr(np, name)(data, axis=axis)
    np.testing.assert_array_equal(
        ak.to_numpy(result) if axis is not None else result, expected
    )
    if axis is not None:
        assert (
            result.layout.form
            == getattr(ak, name)(array.layout.to_typetracer(), axis=axis).layout.form
        )


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble])
def test_order_and_subrange_equality(dtype):
    data = np.array([3, 1, 2, 1], dtype=dtype)
    array = ak.Array(data)
    for stable in (False, True):
        np.testing.assert_array_equal(
            ak.to_numpy(ak.sort(array, stable=stable)), np.sort(data)
        )
    assert ak.to_list(ak.argsort(array, stable=True)) == [1, 3, 2, 0]
    assert ak.to_list(ak._do.unique(array.layout, axis=None)) == [1, 2, 3]
    repeated = ak.contents.NumpyArray(np.array([1, 2, 1, 2, 1, 2], dtype=dtype))
    assert repeated._subranges_equal(
        ak.index.Index64(np.array([0, 2, 4])), ak.index.Index64(np.array([2, 4, 6])), 2
    )


def test_complex_unique_and_sort_errors():
    for dtype in (np.complex64, np.complex128, np.clongdouble):
        array = ak.Array(np.array([2 + 1j, 1 + 3j, 2 + 1j], dtype=dtype))
        assert ak.to_list(ak._do.unique(array.layout, axis=None)) == [1 + 3j, 2 + 1j]
        for sort in (ak.sort, ak.argsort):
            with pytest.raises(TypeError, match="not supported"):
                sort(array)
        with pytest.raises(TypeError, match="unique along an axis"):
            ak._do.unique(ak.unflatten(array, [2, 1]).layout, axis=-1)


def test_extended_initial_precision():
    if np.finfo(np.longdouble).nmant == np.finfo(np.float64).nmant:
        pytest.skip("requires extended precision")
    one = np.longdouble(1)
    eps = np.finfo(np.longdouble).eps
    data = np.array([[one + 2 * eps, one + 3 * eps]], dtype=np.longdouble)
    array = ak.Array(data)
    initial = one + eps
    assert ak.to_numpy(ak.min(array, axis=-1, initial=initial))[0] == initial
    assert (
        ak.to_numpy(ak.max(array, axis=-1, initial=one + 4 * eps))[0] == one + 4 * eps
    )
    assert ak.to_numpy(ak.sum(array, axis=-1))[0] == np.sum(data, axis=-1)[0]
    np.testing.assert_array_equal(ak.to_numpy(ak.sort(array)), data)


@pytest.mark.skipif(
    sys.platform == "emscripten",
    reason="Emscripten does not support subprocess execution",
)
def test_numpy_complex_scalar_ingestion_is_safe():
    # A regression must report a failed subprocess rather than crash pytest.
    code = """
import awkward as ak
import numpy as np
for dtype in (np.complex64, np.complex128, np.clongdouble):
    scalar = dtype(1 + 2j)
    builder = ak.ArrayBuilder()
    builder.append(scalar)
    assert builder.snapshot().to_list() == [1 + 2j]
    assert ak.from_iter([scalar]).to_list() == [1 + 2j]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=dict(os.environ), timeout=30
    )


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble])
def test_json(dtype):
    array = ak.Array(np.array([1.5, np.nan, np.inf, -np.inf], dtype=dtype))
    assert json.loads(
        ak.to_json(array, nan_string="nan", posinf_string="inf", neginf_string="-inf")
    ) == [1.5, "nan", "inf", "-inf"]
    complex_array = ak.Array(np.array([1.5 + 2.5j], dtype=np.clongdouble))
    assert json.loads(ak.to_json(complex_array, complex_record_fields=("r", "i"))) == [
        {"r": 1.5, "i": 2.5}
    ]


@pytest.mark.parametrize(
    "dtype,key",
    [
        (np.float16, "float"),
        (np.longdouble, "longfloat"),
        (np.clongdouble, "longcomplexfloat"),
    ],
)
def test_formatter(dtype, key):
    stream = io.StringIO()
    ak.Array(np.array([1.5], dtype=dtype)).show(
        formatter={key: lambda _: "value"}, stream=stream
    )
    assert stream.getvalue() == "[value]\n"


def test_integration_type_errors():
    from awkward._connect import cling

    form = ak.forms.NumpyForm("float16")
    with pytest.raises(TypeError, match="float16"):
        cling.togenerator(form, flatlist_as_rvec=False).value_type()
    if hasattr(np, "float128"):
        assert (
            cling.togenerator(
                ak.forms.NumpyForm("float128"), flatlist_as_rvec=False
            ).value_type()
            == "long double"
        )
        assert (
            cling.togenerator(
                ak.forms.NumpyForm("complex256"), flatlist_as_rvec=False
            ).value_type()
            == "std::complex<long double>"
        )
    pytest.importorskip("pyarrow")
    with pytest.raises(TypeError, match="Arrow has no type for complex"):
        ak.to_arrow(ak.Array(np.array([1j], dtype=np.clongdouble)))


def test_numba_half_read_error():
    numba = pytest.importorskip("numba")
    ak.numba.register_and_check()

    @numba.njit
    def first(array):
        return array[0]

    with pytest.raises(TypeError, match="Numba cannot read float16"):
        first(ak.Array(np.array([1], dtype=np.float16)))


def test_covariance_dtype():
    x = ak.Array(np.array([[1, 2, 4, 8]], dtype=np.longdouble))
    y = ak.Array(np.array([[1, 3, 2, 9]], dtype=np.longdouble))
    result = ak.to_numpy(ak.covar(x, y, axis=-1))
    assert result.dtype == np.dtype(np.longdouble)
    np.testing.assert_allclose(
        result[0], np.cov(ak.to_numpy(x)[0], ak.to_numpy(y)[0], bias=True)[0, 1]
    )
    assert ak.to_numpy(ak.corr(x, y, axis=-1)).dtype == np.dtype(np.longdouble)


def test_extended_range_and_complex_components():
    if np.finfo(np.longdouble).maxexp <= np.finfo(np.float64).maxexp:
        pytest.skip("requires extended range")
    large = np.longdouble("1e400")
    tiny = np.longdouble("1e-400")
    data = np.array([[large, large], [tiny, tiny]])
    np.testing.assert_array_equal(
        ak.to_numpy(ak.sum(ak.Array(data), axis=-1)), np.sum(data, axis=-1)
    )
    assert ak.to_list(ak.all(ak.Array(data), axis=-1)) == [True, True]
    assert ak.to_list(ak.count_nonzero(ak.Array(data), axis=-1)) == [2, 2]
    values = np.empty((2, 2), dtype=np.clongdouble)
    values.real = data
    values.imag = data
    np.testing.assert_array_equal(
        ak.to_numpy(ak.sum(ak.Array(values), axis=-1)), np.sum(values, axis=-1)
    )


def test_numba_extended_type_error():
    if not hasattr(np, "float128"):
        pytest.skip("requires an extended primitive")
    numba = pytest.importorskip("numba")
    ak.numba.register_and_check()

    @numba.njit
    def identity(array):
        return array

    for dtype in (np.longdouble, np.clongdouble):
        with pytest.raises(TypeError, match="Numba does not support"):
            identity(ak.Array(np.array([1], dtype=dtype)))


def test_rdataframe_longdouble_roundtrip():
    ROOT = pytest.importorskip("ROOT")
    dataframe = ROOT.RDataFrame(3).Define("x", "1.0L / 3.0L + rdfentry_")
    array = ak.from_rdataframe(dataframe, columns=("x",))["x"]
    values = ak.to_numpy(array)
    assert values.dtype == np.dtype(np.longdouble)
    assert values[0] == np.longdouble(1) / np.longdouble(3)
    back = ak.from_rdataframe(ak.to_rdataframe({"x": array}), columns=("x",))["x"]
    np.testing.assert_array_equal(ak.to_numpy(back), values)


@pytest.mark.parametrize("axis", [None, -1, 0])
def test_extended_statistics_keep_precision(axis):
    if np.finfo(np.longdouble).nmant == np.finfo(np.float64).nmant:
        pytest.skip("requires extended precision")
    one = np.longdouble(1)
    step = np.finfo(np.longdouble).eps * 8
    data = one + np.array([[0, 1, 2], [3, 4, 5]], dtype=np.longdouble) * step
    actual = ak.var(ak.Array(data), axis=axis)
    np.testing.assert_allclose(
        ak.to_numpy(actual) if axis is not None else actual,
        np.var(data, axis=axis),
        rtol=1e-12,
        atol=0,
    )
    # The wide value must be raised to its power before any narrowing.
    data = np.array([[np.longdouble("1e200"), np.longdouble("2e200")]])
    expected = np.mean(data**2, axis=-1)
    result = ak.to_numpy(ak.moment(ak.Array(data), 2, axis=-1))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)

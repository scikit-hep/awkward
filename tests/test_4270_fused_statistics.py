# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE
from __future__ import annotations

import numpy as np
import pytest

import awkward as ak
from awkward import _reducers

DTYPES = [
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
    np.complex64,
    np.complex128,
    np.clongdouble,
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [-3, 0, 1, 2, 3, 5])
def test_fused_moment(dtype, n):
    data = np.array([[1, 2, 3], [2, 3, 4]], dtype=dtype)
    if data.dtype.kind == "c":
        data.imag = [[1, -1, 2], [-2, 1, -1]]
    target = (
        data.dtype if data.dtype.kind == "c" else np.result_type(data.dtype, np.float64)
    )
    expected = np.mean(data.astype(target) ** n, axis=1)
    array = ak.Array(data)
    actual = ak.moment(array, n, axis=-1)
    np.testing.assert_allclose(ak.to_numpy(actual), expected, rtol=2e-6)
    assert (
        actual.layout.form
        == ak.moment(array.layout.to_typetracer(), n, axis=-1).layout.form
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_variance(dtype):
    data = np.array([[1, 2, 4], [3, 4, 8]], dtype=dtype)
    if data.dtype.kind == "c":
        data.imag = [[1, -1, 2], [2, -1, 1]]
    array = ak.Array(data)
    for axis in (None, -1, 0):
        target = (
            data.dtype
            if data.dtype.kind == "c"
            else np.result_type(data.dtype, np.float64)
        )
        expected = np.var(data.astype(target), axis=axis)
        actual = ak.var(array, axis=axis)
        np.testing.assert_allclose(
            ak.to_numpy(actual) if axis is not None else actual, expected, rtol=2e-6
        )
    assert (
        ak.var(array, axis=-1).layout.form
        == ak.var(array.layout.to_typetracer(), axis=-1).layout.form
    )


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        np.longdouble,
        np.complex64,
        np.complex128,
        np.clongdouble,
    ],
)
@pytest.mark.parametrize("kind", ["square", "power", "centered"])
def test_direct_fused_bins(dtype, kind):
    data = np.array([99, 1, 2, 3, 88], dtype=dtype)
    if data.dtype.kind == "c":
        data.imag = [0, 1, -1, 2, 0]
    layout = ak.contents.NumpyArray(data)
    offsets = ak.index.Index64(np.array([1, 3, 3, 4]))
    starts = ak.index.Index64(offsets.data[:-1])
    target = (
        data.dtype if data.dtype.kind == "c" else np.result_type(data.dtype, np.float64)
    )
    means = np.array([1.5, 0, 3], dtype=target)
    reducer = {
        "square": _reducers.SumOfSquares(),
        "power": _reducers.SumOfPowers(3),
        "centered": _reducers.CenteredSumOfSquares(means),
    }[kind]
    result = reducer.apply(layout, offsets, starts, None, 3).data
    values = data.astype(target)
    expected = []
    for i in range(3):
        part = values[offsets[i] : offsets[i + 1]]
        if kind == "centered":
            part = abs(part - means[i]) ** 2
        else:
            part = part ** (2 if kind == "square" else 3)
        expected.append(np.sum(part))
    np.testing.assert_allclose(result, expected, rtol=2e-6)
    assert result[1] == 0


def test_wide_fused_range_and_precision():
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("requires extended precision")
    one = np.longdouble(1)
    step = 8 * np.finfo(np.longdouble).eps
    values = one + np.array([[0, 1, 2], [3, 4, 5]], dtype=np.longdouble) * step
    np.testing.assert_allclose(
        ak.to_numpy(ak.var(ak.Array(values), axis=-1)),
        np.var(values, axis=-1),
        rtol=1e-12,
        atol=0,
    )
    values = np.array([[np.longdouble("1e200"), np.longdouble("2e200")]])
    np.testing.assert_allclose(
        ak.to_numpy(ak.moment(ak.Array(values), 2, axis=-1)),
        np.mean(values**2, axis=-1),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "dtype", [np.complex64, np.complex128, np.clongdouble, np.longdouble]
)
def test_public_statistics_use_fused_kernels(dtype, monkeypatch):
    calls = []
    original = _reducers.CenteredSumOfSquares.apply

    def apply(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_reducers.CenteredSumOfSquares, "apply", apply)
    ak.var(ak.Array(np.array([[1, 2], [3, 4]], dtype=dtype)), axis=-1)
    assert calls


def test_first_power_preserves_nonfinite_components():
    values = np.array([[complex(np.inf, 1)], [complex(2, np.nan)]])
    content = ak.contents.NumpyArray(values.reshape(-1))
    offsets = ak.index.Index64(np.array([0, 1, 2]))
    starts = ak.index.Index64(np.array([0, 1]))
    actual = _reducers.SumOfPowers(1).apply(content, offsets, starts, None, 2).data
    np.testing.assert_array_equal(actual.real, values[:, 0].real)
    np.testing.assert_array_equal(actual.imag, values[:, 0].imag)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.clongdouble])
@pytest.mark.parametrize("n", [-3, -1, 0, 1, 2, 3])
def test_complex_power_matches_reference(dtype, n, power_reference):
    namespace = power_reference
    values = np.array(
        [0j, 1 + 2j, complex(np.inf, 1), complex(np.inf, np.inf), complex(1, np.nan)],
        dtype=dtype,
    )
    offsets = ak.index.Index64(np.arange(len(values) + 1, dtype=np.int64))
    starts = ak.index.Index64(offsets.data[:-1])
    result = (
        _reducers.SumOfPowers(n)
        .apply(ak.contents.NumpyArray(values), offsets, starts, None, len(values))
        .data
    )
    expected = np.zeros_like(result)
    with np.errstate(all="ignore"):
        namespace["awkward_reduce_sumofpowers_complex"](
            expected.view(expected.real.dtype),
            values.view(values.real.dtype),
            offsets.data,
            len(values),
            n,
        )
    np.testing.assert_allclose(result.real, expected.real, rtol=3e-6, equal_nan=True)
    np.testing.assert_allclose(result.imag, expected.imag, rtol=3e-6, equal_nan=True)


@pytest.fixture(scope="module")
def power_reference():
    import textwrap
    from pathlib import Path

    specification = (
        Path(__file__).resolve().parents[1] / "kernel-specification.yml"
    ).read_text()
    start = specification.index("      def awkward_reduce_sumofpowers_complex(")
    stop = specification.index("    automatic-tests:", start)
    definition = textwrap.dedent(specification[start:stop])
    namespace = {}
    exec(definition, namespace)
    return namespace


def test_centered_mean_can_widen_real_input():
    content = ak.contents.NumpyArray(np.array([1, 2, 3], dtype=np.float64))
    offsets = ak.index.Index64(np.array([0, 3]))
    starts = ak.index.Index64(np.array([0]))
    means = np.array([2], dtype=np.longdouble)
    result = (
        _reducers.CenteredSumOfSquares(means)
        .apply(content, offsets, starts, None, 1)
        .data
    )
    assert result.dtype == np.dtype(np.longdouble)
    assert result[0] == 2

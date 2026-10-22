# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import itertools

import numpy as np
import pytest

import awkward as ak
from awkward import _reducers
from awkward.contents.numpyarray import _reduce_extended

VALUES = [
    -1 + 0j,
    1 + 0j,
    complex(0, np.nan),
    complex(np.nan, 0),
    complex(np.nan, np.nan),
    complex(1, np.nan),
    complex(np.inf, -np.inf),
]
ROWS = [[], *[[x] for x in VALUES], *map(list, itertools.product(VALUES, repeat=2))]
ROWS += [[1, complex(0, np.nan), 2], [1 + 2j, 1 + 1j, 1 + 1j]]


def assert_same_components(actual, expected):
    # Check the components independently: complex NaN equality alone can hide
    # a changed finite component alongside a NaN.
    np.testing.assert_array_equal(actual.real, expected.real)
    np.testing.assert_array_equal(actual.imag, expected.imag)


@pytest.mark.parametrize(
    "reducer",
    [
        _reducers.Min(None),
        _reducers.Max(None),
        _reducers.Min(0),
        _reducers.Max(0),
        _reducers.ArgMin(),
        _reducers.ArgMax(),
    ],
)
@pytest.mark.parametrize("shifted", [False, True])
def test_extended_complex_nan_path_matches_complex_kernels(reducer, shifted):
    # Exercise the extended reducer helper even on platforms without complex256.
    layout = ak.Array(ROWS).layout
    array = layout.content
    offsets = layout.offsets
    starts = ak.index.Index64(offsets.data[:-1].copy())
    shifts = (
        ak.index.Index64(np.arange(array.length, dtype=np.int64)) if shifted else None
    )
    expected = reducer.apply(array, offsets, starts, shifts, layout.length).data
    actual = _reduce_extended(
        reducer, array, offsets, starts, shifts, layout.length
    ).data
    assert actual.dtype == expected.dtype
    assert_same_components(actual, expected)


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="no complex256 on this platform"
)
@pytest.mark.parametrize("op", [ak.min, ak.max, ak.argmin, ak.argmax])
@pytest.mark.parametrize("axis", [None, -1])
@pytest.mark.parametrize("mask_identity", [False, True])
def test_complex256_nan_matches_complex128(op, axis, mask_identity):
    array = ak.Array(ROWS)
    extended = ak.values_astype(array, np.complex256)
    expected = op(array, axis=axis, mask_identity=mask_identity)
    actual = op(extended, axis=axis, mask_identity=mask_identity)
    if axis is None:
        assert_same_components(np.asarray(actual), np.asarray(expected))
    else:
        assert ak.to_list(ak.is_none(actual)) == ak.to_list(ak.is_none(expected))
        assert_same_components(
            ak.to_numpy(ak.fill_none(actual, 0)),
            ak.to_numpy(ak.fill_none(expected, 0)),
        )


@pytest.mark.skipif(
    not hasattr(np, "complex256"), reason="no complex256 on this platform"
)
@pytest.mark.parametrize("component", ["real", "imag"])
def test_complex256_nan_preserves_extended_precision(component):
    one = np.longdouble(1)
    larger = one + np.longdouble(2) ** -60
    data = np.ones(3, dtype=np.complex256)
    getattr(data, component)[:] = [larger, one, one]
    # The last value must not displace either extremum through a real NaN.
    data.real[2] = np.nan
    array = ak.Array(data.reshape(1, -1))
    assert ak.argmin(array, axis=-1)[0] == 1
    assert ak.argmax(array, axis=-1)[0] == 0
    assert_same_components(ak.to_numpy(ak.min(array, axis=-1)), data[1:2])
    assert_same_components(ak.to_numpy(ak.max(array, axis=-1)), data[0:1])

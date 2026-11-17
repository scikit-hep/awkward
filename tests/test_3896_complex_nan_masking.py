# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128, np.clongdouble])
def test_nan_to_none_masks_either_complex_component(dtype):
    data = np.array([complex(np.nan, 3), 1 + 2j, complex(4, np.nan)], dtype=dtype)
    layout = ak.contents.NumpyArray(data)
    indexed = ak.contents.IndexedOptionArray(
        ak.index.Index64(np.array([0, -1, 1, 2])), layout
    )
    assert ak.to_list(ak.nan_to_none(indexed)) == [None, None, 1 + 2j, None]
    actual = ak.nan_to_none(indexed, highlevel=False)
    traced = ak.nan_to_none(indexed.to_typetracer(forget_length=True), highlevel=False)
    assert traced.form == actual.form


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("axis", [None, 0, -1])
@pytest.mark.parametrize(
    "operation",
    [
        "nansum",
        "nanprod",
        "nanmean",
        "nanvar",
        "nanstd",
        "nanmin",
        "nanmax",
        "nanargmin",
        "nanargmax",
    ],
)
def test_complex_nan_reducers_match_numpy(dtype, axis, operation):
    data = np.array(
        [[1 + 2j, complex(3, np.nan), 0j], [complex(np.nan, 4), 3 + 4j, 2j]],
        dtype=dtype,
    )
    expected = getattr(np, operation)(data, axis=axis)
    actual = getattr(ak, operation)(data, axis=axis)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)

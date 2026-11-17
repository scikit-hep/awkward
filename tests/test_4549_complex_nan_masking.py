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


@pytest.mark.parametrize("operation", [ak.nansum, ak.nanmean])
@pytest.mark.parametrize("mask_identity", [False, True])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("axis", [0, -1, None])
def test_all_valid_complex_reduction_preserves_masked_semantics(
    operation, mask_identity, keepdims, axis
):
    array = ak.Array([[1 + 2j, 3j], [], [5 + 2j]])
    options = {"axis": axis, "mask_identity": mask_identity, "keepdims": keepdims}
    ordinary = ak.sum if operation is ak.nansum else ak.mean
    expected = ordinary(ak.nan_to_none(array), **options)
    actual = operation(array, **options)
    if isinstance(actual, ak.Array):
        assert actual.layout.form == expected.layout.form
        actual_numpy, expected_numpy = ak.to_numpy(actual), ak.to_numpy(expected)
        np.testing.assert_array_equal(
            np.ma.getmaskarray(actual_numpy), np.ma.getmaskarray(expected_numpy)
        )
        np.testing.assert_allclose(
            np.ma.compressed(actual_numpy), np.ma.compressed(expected_numpy)
        )
        traced = operation(
            ak.Array(array.layout.to_typetracer(forget_length=True)), **options
        )
        assert traced.layout.form == actual.layout.form
    else:
        assert actual == expected


@pytest.mark.parametrize("operation", [ak.nansum, ak.nanmean])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("mask_identity", [False, True])
def test_dense_complex_reduction_without_compaction(
    operation, dtype, keepdims, mask_identity
):
    array = ak.Array(
        np.tile(
            np.array(
                [[1j, 2j], [complex(3, np.nan), 4j], [complex(np.nan, 2)] * 2],
                dtype=dtype,
            ),
            (1, 64),
        )
    )
    options = {"axis": -1, "keepdims": keepdims, "mask_identity": mask_identity}
    ordinary = ak.sum if operation is ak.nansum else ak.mean
    actual = operation(array, **options)
    expected = ordinary(ak.nan_to_none(array), **options)
    traced = operation(
        ak.Array(array.layout.to_typetracer(forget_length=True)), **options
    )
    assert actual.layout.form == expected.layout.form == traced.layout.form
    actual_numpy, expected_numpy = ak.to_numpy(actual), ak.to_numpy(expected)
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual_numpy), np.ma.getmaskarray(expected_numpy)
    )
    np.testing.assert_allclose(
        np.ma.compressed(actual_numpy), np.ma.compressed(expected_numpy)
    )

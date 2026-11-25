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


@pytest.mark.parametrize("kind", ["regular", "offsets"])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("operation", [ak.nansum, ak.nanmean])
@pytest.mark.parametrize("mask_identity", [False, True])
@pytest.mark.parametrize("pattern", ["finite", "nan", "infinity"])
def test_matrix_layout_fast_paths(kind, dtype, operation, mask_identity, pattern):
    data = np.tile(np.array([[1 + 2j, -1j, 3 - 4j]], dtype=dtype), (4, 8))
    if pattern == "nan":
        data[0, 0] = complex(3, np.nan)
        data[1] = complex(np.nan, 4)
    elif pattern == "infinity":
        data[0, 0] = complex(np.inf, 2)
        data[0, 1] = complex(-np.inf, 3)
    # Unused prefixes and tails must not enter either reduction.
    backing = np.concatenate(
        [np.array([999j], dtype=dtype), data.ravel(), np.array([999j], dtype=dtype)]
    )
    offsets = np.arange(1, data.size + 2, data.shape[1], dtype=np.int64)
    layout = ak.contents.ListOffsetArray(
        ak.index.Index64(offsets), ak.contents.NumpyArray(backing)
    )
    if kind == "regular":
        layout = layout.to_RegularArray()
    array = ak.Array(layout, attrs={"source": "test"})
    options = {"axis": 1, "mask_identity": mask_identity, "keepdims": True}
    with np.errstate(invalid="ignore"):
        result = operation(array, **options)
        ordinary = ak.sum if operation is ak.nansum else ak.mean
        expected = ordinary(ak.nan_to_none(array), **options)
    assert result.attrs["source"] == "test"
    assert result.layout.form == expected.layout.form
    actual_numpy, expected_numpy = ak.to_numpy(result), ak.to_numpy(expected)
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual_numpy), np.ma.getmaskarray(expected_numpy)
    )
    np.testing.assert_allclose(
        np.ma.compressed(actual_numpy), np.ma.compressed(expected_numpy), equal_nan=True
    )
    traced = operation(ak.Array(layout.to_typetracer(forget_length=True)), **options)
    assert traced.layout.form == result.layout.form


@pytest.mark.parametrize("kind", ["numpy", "regular", "offsets"])
@pytest.mark.parametrize("operation", [ak.nansum, ak.nanmean])
@pytest.mark.parametrize("pattern", ["finite", "nan", "infinity"])
def test_short_row_sum_validation(kind, operation, pattern):
    data = np.full((128, 64), 1 + 2j, dtype=np.complex128)
    if pattern == "nan":
        data[0, 0] = complex(3, np.nan)
        data[1] = complex(np.nan, 2)
    elif pattern == "infinity":
        data[0, 0] = complex(np.inf, 2)
        data[0, 1] = complex(-np.inf, 1)
    if kind == "numpy":
        array = ak.Array(data)
    else:
        layout = ak.contents.ListOffsetArray(
            ak.index.Index64(np.arange(0, data.size + 1, 64, dtype=np.int64)),
            ak.contents.NumpyArray(data.ravel()),
        )
        array = ak.Array(layout.to_RegularArray() if kind == "regular" else layout)
    with np.errstate(invalid="ignore"):
        result = operation(array, axis=-1, mask_identity=True)
        ordinary = ak.sum if operation is ak.nansum else ak.mean
        expected = ordinary(ak.nan_to_none(array), axis=-1, mask_identity=True)
    actual_numpy, expected_numpy = ak.to_numpy(result), ak.to_numpy(expected)
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual_numpy), np.ma.getmaskarray(expected_numpy)
    )
    np.testing.assert_allclose(
        np.ma.compressed(actual_numpy), np.ma.compressed(expected_numpy), equal_nan=True
    )
    assert result.layout.form == expected.layout.form


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("kind", ["numpy", "regular", "offsets"])
@pytest.mark.parametrize("shape", [(2, 127), (2, 128), (63, 64), (64, 64), (2, 100000)])
@pytest.mark.parametrize("operation", [ak.nansum, ak.nanmean])
@pytest.mark.parametrize("has_nan", [False, True])
@pytest.mark.parametrize("mask_identity", [False, True])
@pytest.mark.parametrize("keepdims", [False, True])
def test_matrix_accumulation_matches_masked_reduction(
    dtype, kind, shape, operation, has_nan, mask_identity, keepdims
):
    data = np.full(shape, 0.1 + 0.1j, dtype=dtype)
    if has_nan:
        data[0, ::3] = complex(1, np.nan)
        data[1] = complex(np.nan, 1)
    if kind == "numpy":
        array = ak.Array(data)
    else:
        # Exercise a nonzero offset and unused prefix/tail as well.
        backing = np.concatenate(([999j], data.ravel(), [999j])).astype(dtype)
        layout = ak.contents.ListOffsetArray(
            ak.index.Index64(np.arange(1, data.size + 2, shape[1])),
            ak.contents.NumpyArray(backing),
        )
        array = ak.Array(layout.to_RegularArray() if kind == "regular" else layout)
    options = {"axis": -1, "keepdims": keepdims, "mask_identity": mask_identity}
    ordinary = ak.sum if operation is ak.nansum else ak.mean
    expected = ordinary(ak.nan_to_none(array), **options)
    actual = operation(array, **options)
    assert actual.layout.form == expected.layout.form
    actual_np, expected_np = ak.to_numpy(actual), ak.to_numpy(expected)
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual_np), np.ma.getmaskarray(expected_np)
    )
    np.testing.assert_array_equal(
        np.ma.compressed(actual_np), np.ma.compressed(expected_np)
    )

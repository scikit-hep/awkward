# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak


@pytest.mark.parametrize("source_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("fill", [0.1, 1e-5, np.float64(0.1)])
def test_float_fill_preserves_precision(source_dtype, fill):
    source = ak.Array(np.array([1], dtype=source_dtype))
    result = ak.to_dense(ak.concatenate([source, ak.Array([None])]), fill_value=fill)
    assert result.dtype == np.float64
    assert result[1] == fill


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize(
    "dtype,fill",
    [(np.uint8, -1), (np.uint8, np.int64(-1)), (np.int64, np.nan), (np.int8, 1000)],
)
def test_invalid_fill_is_rejected_independently_of_missing_data(dtype, fill, missing):
    array = [[1], []] if missing else [[1], [2]]
    with pytest.raises(TypeError):
        ak.to_dense(array, dtype=dtype, fill_value=fill)


def test_explicit_data_cast_follows_numpy():
    source = np.array([300.7])
    np.testing.assert_array_equal(
        ak.to_dense(source, dtype=np.int8), source.astype(np.int8)
    )


def test_typed_fill_participates_in_promotion():
    source = np.array([1], dtype=np.float32)
    assert ak.to_dense(source, fill_value=np.float64(0)).dtype == np.float64
    assert (
        ak.to_dense(source, fill_value=np.array(0, dtype=np.float64)).dtype
        == np.float64
    )
    assert ak.to_dense(source, fill_value=0).dtype == np.result_type(source.dtype, 0)


def test_boolean_default_fill_uses_numpy_promotion():
    source = np.array([True, False])
    assert ak.to_dense(source).dtype == np.result_type(source.dtype, 0)


def test_explicit_complex_to_real_cast_warns():
    warning = getattr(np, "exceptions", np).ComplexWarning
    with pytest.warns(warning):
        result = ak.to_dense([1 + 2j], dtype=np.float64)
    np.testing.assert_array_equal(result, [1])

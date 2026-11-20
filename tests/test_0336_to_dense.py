# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pytestmark = pytest.mark.filterwarnings("ignore::awkward.errors.ExperimentalWarning")


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([[1, 2], [], [3]], [[1, 2], [0, 0], [3, 0]]),
        ([[1, None], None, []], [[1, 0], [0, 0], [0, 0]]),
        ([None, None], [0, 0]),
        ([], []),
        ([[], []], [[], []]),
        ([[[1], [2, 3]], [[4]]], [[[1, 0], [2, 3]], [[4, 0], [0, 0]]]),
        ([[[1, None], None], None], [[[1, 0], [0, 0]], [[0, 0], [0, 0]]]),
        ([{"a": 1, "b": 2.5}, None], [[1, 2.5], [0, 0]]),
        ([{"a": {"x": 1, "y": 2}, "b": 3}], [[1, 2, 3]]),
        ([(1, 2), (3, 4)], [[1, 2], [3, 4]]),
        (
            [{"a": [1], "b": [2, 3]}, {"a": [], "b": [4]}],
            [[[1, 2], [0, 3]], [[0, 4], [0, 0]]],
        ),
        ([{"a": None, "b": [1, 2]}], [[[0, 1], [0, 2]]]),
    ],
)
def test_padding(data, expected):
    result = ak.to_dense(data)
    np.testing.assert_array_equal(result, expected)
    assert type(result) is np.ndarray
    assert result.flags.c_contiguous
    assert result.dtype.kind in "biufc"


def test_issue_example():
    array = ak.zip(
        {"a": [[1, 2], [], [3, 4, 5]], "b": [[1.1, 2.2], [], [3.3, 4.4, 5.5]]}
    )
    np.testing.assert_array_equal(
        ak.to_dense(array),
        [
            [[1, 1.1], [2, 2.2], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
            [[3, 3.3], [4, 4.4], [5, 5.5]],
        ],
    )
    assert ak.to_dense(array, clip=True).shape == (3, 0, 2)


def test_clipping():
    np.testing.assert_array_equal(
        ak.to_dense([[1, 2], None, [3]], clip=True), [[1], [0], [3]]
    )
    np.testing.assert_array_equal(
        ak.to_dense([[[1, 2], [3]], [[4, 5]]], clip=True), [[[1]], [[4]]]
    )


@pytest.mark.parametrize("shape", [(0, 3), (2, 0, 3), (0, 2, 3), (2, 3, 4)])
def test_regular_shapes(shape):
    array = np.arange(np.prod(shape)).reshape(shape)
    result = ak.to_dense(array)
    assert result.shape == shape
    np.testing.assert_array_equal(result, array)


def test_dtype_and_fill():
    array = ak.Array([[1.5], []])
    result = ak.to_dense(array, fill_value=-1, dtype=np.float32)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[1.5], [-1]])
    assert ak.to_dense(np.array([1], dtype=np.float32)).dtype == np.float32
    assert np.isnan(ak.to_dense([1, None], fill_value=np.nan)[1])
    np.testing.assert_array_equal(ak.to_dense([1, None], fill_value=2j), [1, 2j])


@pytest.mark.parametrize("data", [["one"], [b"one"], [{"x": "one"}], [{}]])
def test_nonnumeric(data):
    with pytest.raises(TypeError):
        ak.to_dense(data)


@pytest.mark.parametrize("dtype", [object, "U3", [("x", "i8")]])
def test_invalid_dtype(dtype):
    with pytest.raises(TypeError, match="numeric NumPy dtype"):
        ak.to_dense([1], dtype=dtype)


@pytest.mark.parametrize("fill", [None, "x", [0]])
def test_invalid_fill(fill):
    with pytest.raises(TypeError, match="numeric scalar"):
        ak.to_dense([1], fill_value=fill)


def test_incompatible_dimensions():
    with pytest.raises(ValueError, match="list dimensions"):
        ak.to_dense([1, [2]])
    with pytest.raises(ValueError, match="list dimensions"):
        ak.to_dense([{"x": 1, "y": [2]}])


def test_typetracer():
    with pytest.raises(TypeError, match="concrete data"):
        ak.to_dense(ak.Array([[1]]).layout.to_typetracer(forget_length=True))


def test_sliced_and_masked():
    array = ak.Array([[99], [1, 2], [], [3]])[1:]
    np.testing.assert_array_equal(ak.to_dense(array), [[1, 2], [0, 0], [3, 0]])
    masked = np.ma.array([[1, 2], [3, 4]], mask=[[False, True], [True, False]])
    np.testing.assert_array_equal(ak.to_dense(masked), [[1, 0], [0, 4]])


def test_union_records():
    # Field order follows first appearance, and absent fields are filled.
    array = ak.concatenate([ak.Array([{"a": 1}]), ak.Array([{"b": 2}])])
    np.testing.assert_array_equal(ak.to_dense(array), [[1, 0], [0, 2]])
    # Numeric replacements for whole records apply to each feature.
    array = ak.fill_none(ak.Array([{"a": 1, "b": 2}, None]), -1)
    np.testing.assert_array_equal(ak.to_dense(array), [[1, 2], [-1, -1]])


def test_lowlevel_and_regular_record_fields():
    array = ak.zip(
        {"a": np.arange(6).reshape(2, 3), "b": np.arange(6, 12).reshape(2, 3)},
        depth_limit=1,
    )
    expected = np.stack(
        [np.arange(6).reshape(2, 3), np.arange(6, 12).reshape(2, 3)], axis=-1
    )
    np.testing.assert_array_equal(ak.to_dense(array.layout), expected)

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pytestmark = pytest.mark.filterwarnings("ignore::awkward.errors.ExperimentalWarning")


@pytest.mark.parametrize(
    ("array", "indices", "axis", "expected"),
    [
        (
            [[10, 20, 30], [40, 50]],
            [[[2, 0], [1]], [[1], []]],
            1,
            [[[30, 10], [20]], [[50], []]],
        ),
        (
            [[], [], list(range(7))],
            [[], [], [[2, 5], [6, 2]]],
            1,
            [[], [], [[2, 5], [6, 2]]],
        ),
        ([[1, 2], [3]], [[[-1, -2], [0, 0]], [[-1]]], -1, [[[2, 1], [1, 1]], [[3]]]),
        ([[1, 2], [3]], [1, 0], 1, [2, 3]),
        ([[1, 2], [3]], [[[None, 0], None], [[]]], 1, [[[None, 1], None], [[]]]),
        ([[1, 2], None], [[[0]], [[0]]], 1, [[[1]], None]),
        ([[1, None], [3]], [[[1, 0]], None], 1, [[[None, 1]], None]),
        ([[[1, 2], [3]], [[4]]], [[[1, 0]], [[0]]], 1, [[[[3], [1, 2]]], [[[4]]]]),
        ([1, 2, 3], [[2, 0], [1]], 0, [[3, 1], [2]]),
        ([[], [1]], [[], [[0]]], 1, [[], [[1]]]),
        ([[1], [2]], [[], []], 1, [[], []]),
        ([], [], 0, []),
        ([1, 2], [None, None], 0, [None, None]),
        (
            [[[1, 2], [3]], [[4]]],
            [[[[1], [0]], [[0]]], [[[0]]]],
            2,
            [[[[2], [1]], [[3]]], [[[4]]]],
        ),
        ([[{"x": 1}, {"x": 2}]], [[[1, 0]]], 1, [[[{"x": 2}, {"x": 1}]]]),
        ([["a", "b"]], [[[1, 0]]], 1, [[["b", "a"]]]),
    ],
)
def test_selection(array, indices, axis, expected):
    array = ak.Array(array)
    indices = ak.Array(indices)
    result = ak.broadcast_slice(array, indices, axis)
    assert result.to_list() == expected
    tt = ak.broadcast_slice(
        array.layout.to_typetracer(forget_length=True),
        indices.layout.to_typetracer(forget_length=True),
        axis,
        highlevel=False,
    )
    assert result.layout.form == tt.form


@pytest.mark.parametrize("indices", [[[[2]], [[0]]], [[[-3]], [[0]]]])
def test_bounds(indices):
    with pytest.raises(IndexError):
        ak.broadcast_slice([[1, 2], [3, 4, 5, 6]], indices)


@pytest.mark.parametrize("indices", [[[True]], [[1.5]], [["x"]], [[{"x": 0}]]])
def test_invalid_indices(indices):
    with pytest.raises(TypeError, match="indices must contain integers"):
        ak.broadcast_slice([[1, 2]], indices)


def test_prefix_mismatch():
    with pytest.raises(ValueError, match="same outer length"):
        ak.broadcast_slice([[1, 2]], [[[0]], [[1]]])


@pytest.mark.parametrize("axis", [1, -1])
def test_regular(axis):
    array = np.arange(12).reshape(3, 4)
    indices = np.array([[[3, 0], [1, 2]], [[1, 0], [0, 2]], [[0, 1], [3, 2]]])
    result = ak.broadcast_slice(array, indices, axis)
    assert result.to_list() == [
        [array[i, group].tolist() for group in groups]
        for i, groups in enumerate(indices)
    ]


def test_metadata():
    array = ak.with_name(ak.Array([[{"x": 1}, {"x": 2}]]), "Point")
    result = ak.broadcast_slice(array, [[[1]]], attrs={"source": "test"})
    assert result.attrs == {"source": "test"}
    assert result.layout.purelist_parameter("__record__") == "Point"
    assert isinstance(
        ak.broadcast_slice(array, [[[1]]], highlevel=False), ak.contents.Content
    )


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint8, np.uint64])
def test_integer_dtypes(dtype):
    index = ak.Array(np.array([[[1, 0]], [[0, 0]]], dtype=dtype))
    assert ak.broadcast_slice([[1, 2], [3]], index).to_list() == [[[2, 1]], [[3, 3]]]


def test_sliced_and_indexed_collections():
    source = ak.Array([[99], [10, 20], [30], [40, 50, 60]])
    source = source[[3, 1]][:, ::-1]
    index = ak.Array([[[0, -1]], [[-1, 0]]])
    assert ak.broadcast_slice(source, index).to_list() == [[[60, 40]], [[10, 20]]]


def test_union_payload():
    source = ak.Array([[1, "two"], ["three", 4]])
    index = ak.Array([[[1, 0]], [[1, 0]]])
    assert ak.broadcast_slice(source, index).to_list() == [[["two", 1]], [[4, "three"]]]
    assert (
        ak.broadcast_slice(
            source.layout.to_typetracer(), index.layout.to_typetracer(), highlevel=False
        ).form
        == ak.broadcast_slice(source, index).layout.form
    )


@pytest.mark.parametrize("axis", [2, -3])
def test_invalid_axis(axis):
    with pytest.raises(ak.errors.AxisError):
        ak.broadcast_slice([[1]], [[[0]]], axis)


def test_noninteger_axis():
    with pytest.raises(TypeError, match="axis"):
        ak.broadcast_slice([[1]], [[[0]]], 1.5)


def test_named_axes_are_removed():
    source = ak.with_named_axis([[1, 2]], ("event", "item"))
    result = ak.broadcast_slice(source, [[[0]]])
    assert result.named_axis == {}


@pytest.mark.parametrize("forget_source", [True, False])
@pytest.mark.parametrize("forget_index", [True, False])
def test_typetracer_lengths(forget_source, forget_index):
    source = ak.Array([[1, 2], [3]])
    index = ak.Array([[[0]], [[0]]])
    out = ak.broadcast_slice(
        source.layout.to_typetracer(forget_length=forget_source),
        index.layout.to_typetracer(forget_length=forget_index),
        highlevel=False,
    )
    assert out.form == ak.broadcast_slice(source, index).layout.form


def test_empty_regular_dimensions():
    source = np.empty((0, 4), dtype=np.int64)
    index = np.empty((0, 2, 0), dtype=np.int64)
    result = ak.broadcast_slice(source, index)
    assert result.to_list() == []
    assert ak.to_numpy(result).shape == (0, 2, 0)

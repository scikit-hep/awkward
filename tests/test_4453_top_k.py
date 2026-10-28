# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pytestmark = pytest.mark.filterwarnings("ignore::awkward.errors.ExperimentalWarning")


def check(array, k, axis, values, indices, mode="largest"):
    array = ak.Array(array)
    result = ak.top_k(array, k, axis=axis, mode=mode)
    assert result._fields == ("values", "indices")
    assert result.values.to_list() == values
    assert result.indices.to_list() == indices
    assert ak.top_k_values(array, k, axis=axis, mode=mode).to_list() == values
    assert ak.top_k_indices(array, k, axis=axis, mode=mode).to_list() == indices
    traced = ak.top_k(
        array.layout.to_typetracer(forget_length=True),
        k,
        axis=axis,
        mode=mode,
        highlevel=False,
    )
    assert result.values.layout.form == traced.values.form
    assert result.indices.layout.form == traced.indices.form


@pytest.mark.parametrize("k", [0, 1, 2, 5, np.int64(2)])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
def test_ragged_innermost(k, mode):
    array = [[3, 1, 5, 3], [], [2, 4], [9]]
    values, indices = [], []
    for row in array:
        order = sorted(range(len(row)), key=row.__getitem__, reverse=mode == "largest")[
            :k
        ]
        values.append([row[i] for i in order])
        indices.append(order)
    check(array, k, -1, values, indices, mode)


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("mode", ["largest", "smallest"])
@pytest.mark.parametrize("k", [0, 1, 2, 8])
def test_rectangular(axis, mode, k):
    array = np.array([[[3, 1, 5], [2, 4, 0]], [[9, 8, 7], [6, 2, 1]]])
    order = np.argsort(-array if mode == "largest" else array, axis=axis, kind="stable")
    selection = (slice(None),) * (axis % array.ndim) + (slice(None, k),)
    indices = order[selection]
    values = np.take_along_axis(array, indices, axis=axis)
    check(array, k, axis, values.tolist(), indices.tolist(), mode)


def test_ragged_outer_axes():
    check([[3], [2, 4], [0, 1]], 1, 0, [[3, 4]], [[0, 1]])
    check(
        [[3], [2, 4], [0, 1]],
        3,
        0,
        [[3, 4], [2, 1], [0, None]],
        [[0, 1], [1, 2], [2, None]],
    )
    check([[[1], [2, 3]], [[4]]], 1, 1, [[[2, 3]], [[4]]], [[[1, 1]], [[0]]])
    check(
        [[[1], [2, 3]], [[4]]],
        2,
        0,
        [[[4], [2, 3]], [[1], [None, None]]],
        [[[1], [0, 0]], [[0], [None, None]]],
    )
    check(
        [[[[1]], [[2, 3]]], [[[4, 5, 6]]]],
        1,
        1,
        [[[[2, 3]]], [[[4, 5, 6]]]],
        [[[[1, 1]]], [[[0, 0, 0]]]],
    )


def test_missing():
    check(
        [[3, None, 1], None, []],
        3,
        -1,
        [[3, 1, None], None, []],
        [[0, 2, None], None, []],
    )
    check(
        [[[1], None], [[2, 3]]], 2, -1, [[[1], None], [[3, 2]]], [[[0], None], [[1, 0]]]
    )
    check([[3], None, [2, 1]], 2, 0, [[3, 1], [2, None]], [[0, 2], [2, None]])


@pytest.mark.parametrize("array,axis", [([], 0), ([[], []], 0), ([[], []], 1)])
def test_empty(array, axis):
    check(array, 2, axis, array, array)


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int8, np.uint64, np.float32, np.float64]
)
def test_dtypes(dtype):
    values = np.array([1, 0, 1], dtype=dtype)
    result = ak.top_k(values, 2)
    assert ak.to_numpy(result.values).dtype == values.dtype
    assert ak.to_numpy(result.indices).dtype == np.int64
    np.testing.assert_array_equal(result.values, [1, 1])
    np.testing.assert_array_equal(result.indices, [0, 2])


@pytest.mark.parametrize("k", [True, np.bool_(False), 1.5, None])
def test_invalid_k_type(k):
    with pytest.raises(TypeError, match="k must be"):
        ak.top_k([1, 2], k)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="k must be"):
        ak.top_k([1], -1)
    with pytest.raises(ValueError, match="mode must be"):
        ak.top_k([1], 1, mode="biggest")
    with pytest.raises(TypeError, match="axis"):
        ak.top_k([1], 1, axis=None)
    with pytest.raises(ak.errors.AxisError):
        ak.top_k([1], 1, axis=1)


@pytest.mark.parametrize(
    "array",
    [
        [1j],
        ["one"],
        [b"one"],
        [{"x": 1}],
        np.array(["2020-01-01"], dtype="datetime64[D]"),
    ],
)
def test_invalid_data(array):
    with pytest.raises(TypeError, match="real numeric"):
        ak.top_k(array, 1)


def test_named_axis_and_metadata():
    array = ak.with_named_axis([[1, 3, 2]], ("event", "item"))
    behavior = {}
    result = ak.top_k(
        array,
        2,
        axis="item",
        behavior=behavior,
        attrs={"note": "test", "__named_axis__": array.named_axis},
    )
    assert result.values.to_list() == [[3, 2]]
    for value in result:
        assert value.named_axis == array.named_axis
        assert value.attrs["note"] == "test"
        assert value.behavior is behavior
    result = ak.top_k(array, 2, highlevel=False)
    assert all(isinstance(value, ak.contents.Content) for value in result)


def test_nan_values_indices_correspond():
    array = np.array([1.0, np.nan, 3.0, -np.inf, np.inf])
    result = ak.top_k(array, 5)
    np.testing.assert_allclose(
        result.values, array[ak.to_numpy(result.indices)], equal_nan=True
    )


def test_unsigned_extremes():
    array = np.array([0, 2**64 - 1, 2**63], dtype=np.uint64)
    result = ak.top_k(array, 2)
    assert result.values.to_list() == [2**64 - 1, 2**63]
    assert result.indices.to_list() == [1, 2]


def test_float16():
    array = np.array([1.0, 2.0, 1.5], dtype=np.float16)
    result = ak.top_k(array, 2)
    assert ak.to_numpy(result.values).dtype == np.float16
    assert result.values.to_list() == [2.0, 1.5]


def test_sliced_layout():
    check(ak.Array([[99], [3, 1], [2, 4]])[1:], 1, 0, [[3, 4]], [[0, 1]])


def test_dispatch():
    class Custom:
        def __awkward_function__(self, func, array_likes, args, kwargs):
            return func.__name__

    for function in (ak.top_k, ak.top_k_values, ak.top_k_indices):
        assert function(Custom(), 1) == function.__name__


def _reference_group(rows, rank, k, largest):
    length = min(k, len(rows))
    if rank == 1:
        present = [(value, i) for i, value in enumerate(rows) if value is not None]
        present.sort(key=lambda pair: pair[0], reverse=largest)
        present = present[:length] + [(None, None)] * max(0, length - len(present))
        return [value for value, _ in present], [index for _, index in present]
    width = max((len(row) for row in rows if row is not None), default=0)
    columns = [
        _reference_group(
            [row[j] if row is not None and j < len(row) else None for row in rows],
            rank - 1,
            k,
            largest,
        )
        for j in range(width)
    ]
    return (
        [[column[0][i] for column in columns] for i in range(length)],
        [[column[1][i] for column in columns] for i in range(length)],
    )


def _reference(data, rank, axis, k, largest):
    if axis == 0:
        return _reference_group(data, rank, k, largest)
    result = [
        _reference(row, rank - 1, axis - 1, k, largest)
        if row is not None
        else (None, None)
        for row in data
    ]
    return [item[0] for item in result], [item[1] for item in result]


@pytest.mark.parametrize("rank", [3, 4])
def test_ragged_reference(rank):
    import random

    rng = random.Random(554 + rank)

    def generate(depth):
        if depth == 0:
            return None if rng.random() < 0.2 else rng.randrange(5)
        return [
            generate(depth - 1) if rng.random() > 0.15 else None
            for _ in range(rng.randrange(4))
        ]

    for _ in range(15):
        data = generate(rank)
        array = ak.Array(data)
        if array.ndim != rank:
            continue
        for axis in range(rank):
            k = rng.randrange(5)
            largest = bool(rng.randrange(2))
            values, indices = _reference(data, rank, axis, k, largest)
            check(array, k, axis, values, indices, "largest" if largest else "smallest")


def test_nonlocal_argsort_typetracer_shape():
    array = ak.Array([[3], [2, 4]])
    expected = ak.argsort(array, axis=0)
    traced = ak.argsort(
        array.layout.to_typetracer(forget_length=True), axis=0, highlevel=False
    )
    assert expected.layout.form == traced.form


def test_indexed_layout():
    array = ak.Array([[3], [2, 4], [0, 1]])[[2, 0, 1]]
    check(array, 1, 0, [[3, 4]], [[1, 2]])


def test_signed_extremes():
    values = np.array([-(2**63), 2**63 - 1, 0], dtype=np.int64)
    result = ak.top_k(values, 2)
    assert result.values.to_list() == [2**63 - 1, 0]
    assert result.indices.to_list() == [1, 2]

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

pytestmark = pytest.mark.filterwarnings("ignore::awkward.errors.ExperimentalWarning")


def test_issue_examples():
    array = ak.Array([[(1, 2), (1, 3), (2, 3)], [], [(4, 5), (6, 7, 8)]])
    assert ak.tuples_to_lists(array).to_list() == [
        [[1, 2], [1, 3], [2, 3]],
        [],
        [[4, 5], [6, 7, 8]],
    ]
    assert ak.tuples_to_lists(array, pad=True).to_list() == [
        [[1, 2, None], [1, 3, None], [2, 3, None]],
        [],
        [[4, 5, None], [6, 7, 8]],
    ]
    array = ak.Array([[[1, 2], [1, 3], [2, 3]], [], [[4, 5], [6]]])
    assert ak.lists_to_tuples(array).to_list() == [
        [(1, 2), (1, 3), (2, 3)],
        [],
        [(4, 5), (6, None)],
    ]


def test_named_records_are_preserved():
    array = ak.Array([{"x": (2, 3), "y": 1}, {"x": (4, 5), "y": 0}])
    assert ak.tuples_to_lists(array).to_list() == [
        {"x": [2, 3], "y": 1},
        {"x": [4, 5], "y": 0},
    ]
    assert ak.lists_to_tuples(ak.tuples_to_lists(array)).to_list() == array.to_list()
    array = ak.Array([{"x": (1,), "y": (2, 3)}, {"x": (4, 5, 6), "y": (7, 8)}])
    assert ak.tuples_to_lists(array, pad=True).to_list() == [
        {"x": [1, None, None], "y": [2, 3]},
        {"x": [4, 5, 6], "y": [7, 8]},
    ]
    array = ak.Array([{"x": [1], "y": [2, 3]}, {"x": [4, 5, 6], "y": [7, 8]}])
    assert ak.lists_to_tuples(array).to_list() == [
        {"x": (1, None, None), "y": (2, 3)},
        {"x": (4, 5, 6), "y": (7, 8)},
    ]


@pytest.mark.parametrize("drop_none", [True, False])
def test_missing_tuples_and_fields(drop_none):
    array = ak.Array([(None, 1), None, (2, None)])
    expected = [[1], None, [2]] if drop_none else [[None, 1], None, [2, None]]
    result = ak.tuples_to_lists(array, drop_none=drop_none, pad=not drop_none)
    assert result.to_list() == expected
    assert ak.validity_error(result) == ""
    assert ak.lists_to_tuples(result).to_list() == (
        [(1,), None, (2,)] if drop_none else array.to_list()
    )


@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize("size", [None, 3])
def test_numpy_buffers(dtype, size):
    array = ak.Array(np.arange(24, dtype=dtype).reshape(2, 3, 4))
    result = ak.lists_to_tuples(array, size=size)
    expected = [
        [
            tuple(range(i, i + (4 if size is None else size)))
            for i in range(j, j + 12, 4)
        ]
        for j in [0, 12]
    ]
    assert result.to_list() == expected
    assert ak.validity_error(result) == ""


def test_axes():
    array = ak.Array([[(1, 2)], [], [(3, 4), (5, 6)]])
    assert (
        ak.tuples_to_lists(array, axis=2).to_list()
        == ak.tuples_to_lists(array, axis=-1).to_list()
    )
    lists = ak.tuples_to_lists(array)
    assert ak.lists_to_tuples(lists, axis=2).to_list() == array.to_list()
    assert ak.lists_to_tuples(lists, axis=-1).to_list() == array.to_list()
    assert ak.lists_to_tuples([[1, 2], [3]], axis=1).to_list() == [(1, 2), (3, None)]
    assert ak.lists_to_tuples([[[1], [2]], [[3]]], axis=1).to_list() == [
        ([1], [2]),
        ([3], None),
    ]
    tuples = ak.Array([([1, 2], [3]), ([], [4, 5])])
    assert ak.tuples_to_lists(tuples, axis=1).to_list() == [[[1, 2], [3]], [[], [4, 5]]]
    for operation, axis in [
        (ak.tuples_to_lists, 4),
        (ak.tuples_to_lists, -4),
        (ak.lists_to_tuples, 0),
        (ak.lists_to_tuples, 4),
    ]:
        with pytest.raises(ak.errors.AxisError):
            operation(array, axis=axis)


@pytest.mark.parametrize("size", [None, 0, 2])
def test_empty_lists(size):
    result = ak.lists_to_tuples([[], None, []], size=size)
    width = 0 if size is None else size
    assert result.to_list() == [tuple([None] * width), None, tuple([None] * width)]
    assert ak.validity_error(result) == ""
    assert ak.tuples_to_lists([(), ()]).to_list() == [[], []]
    assert ak.tuples_to_lists(ak.Array([(), ()])[:0]).to_list() == []
    assert ak.lists_to_tuples(ak.Array([[1, 2]])[:0], size=size).to_list() == []


def test_explicit_size_and_invalid_size():
    assert ak.lists_to_tuples([[1, 2, 3], []], size=2).to_list() == [
        (1, 2),
        (None, None),
    ]
    assert ak.lists_to_tuples([[1, 2, 3], []], size=0).to_list() == [(), ()]
    for size in [-1, 1.5, True, np.bool_(True), "2"]:
        with pytest.raises((TypeError, ValueError)):
            ak.lists_to_tuples([[1]], size=size)


def test_sliced_and_indexed_lists():
    source = ak.Array([[0, 1, 2, 3, 4], [1], [2, 3], [4, 5, 6, 7]])
    assert ak.lists_to_tuples(source[1:3]).to_list() == [(1, None), (2, 3)]
    assert ak.lists_to_tuples(source[[2, 1]]).to_list() == [(2, 3), (1, None)]
    masked = ak.mask(source, [False, True, True, False])
    assert ak.lists_to_tuples(masked).to_list() == [None, (1, None), (2, 3), None]


def test_strings_and_heterogeneous_fields():
    array = ak.Array([("ab", 1), ("c", 2)])
    assert ak.tuples_to_lists(array).to_list() == [["ab", 1], ["c", 2]]
    assert ak.lists_to_tuples(ak.tuples_to_lists(array)).to_list() == array.to_list()
    with pytest.raises(ak.errors.AxisError):
        ak.lists_to_tuples(["ab", "c"])
    array = ak.Array([({"x": 1}, {"x": 2})])
    assert ak.tuples_to_lists(array).to_list() == [[{"x": 1}, {"x": 2}]]


def test_behavior_attrs_and_parameters():
    behavior = {"custom": object()}
    array = ak.Array([(1, 2)], attrs={"source": "example"}, behavior=behavior)
    result = ak.tuples_to_lists(array)
    assert result.behavior is behavior
    assert result.attrs["source"] == "example"
    assert ak.lists_to_tuples(result).attrs["source"] == "example"
    assert isinstance(ak.tuples_to_lists(array, highlevel=False), ak.contents.Content)
    assert isinstance(ak.lists_to_tuples(result, highlevel=False), ak.contents.Content)
    array = ak.with_parameter(ak.with_name(array, "TupleName"), "units", "cm")
    result = ak.tuples_to_lists(array)
    assert ak.parameters(result) == {"units": "cm"}
    assert ak.parameters(ak.lists_to_tuples(result)) == {"units": "cm"}


def test_named_axes():
    array = ak.with_named_axis(
        ak.Array([[(1, 2)], [(3, 4)]]), {"events": 0, "objects": 1}
    )
    result = ak.tuples_to_lists(array, axis=2)
    assert result.named_axis == {"events": 0, "objects": 1}
    restored = ak.lists_to_tuples(
        ak.with_named_axis(result, {"events": 0, "objects": 1, "fields": 2}),
        axis="fields",
    )
    assert restored.named_axis == {"events": 0, "objects": 1}
    assert restored.to_list() == array.to_list()


@pytest.mark.parametrize("forget_length", [False, True])
@pytest.mark.parametrize("drop_none", [False, True])
def test_typetracer_tuples_to_lists(forget_length, drop_none):
    array = ak.Array([[(1, None), (2, 3)], [], [(4, 5, 6)]])
    tracer = ak.Array(array.layout.to_typetracer(forget_length=forget_length))
    result = ak.tuples_to_lists(array, drop_none=drop_none, pad=not drop_none)
    assert (
        ak.tuples_to_lists(tracer, drop_none=drop_none, pad=not drop_none).layout.form
        == result.layout.form
    )


@pytest.mark.parametrize("forget_length", [False, True])
@pytest.mark.parametrize("size", [0, 2, 4])
def test_typetracer_lists_to_tuples(forget_length, size):
    array = ak.Array([[[1, 2], []], [], [[3]]])
    tracer = ak.Array(array.layout.to_typetracer(forget_length=forget_length))
    assert (
        ak.lists_to_tuples(tracer, size=size).layout.form
        == ak.lists_to_tuples(array, size=size).layout.form
    )
    with pytest.raises(ValueError, match="explicit size"):
        ak.lists_to_tuples(tracer)
    regular = ak.Array(np.arange(12).reshape(3, 4))
    regular_tracer = ak.Array(regular.layout.to_typetracer(forget_length=forget_length))
    assert (
        ak.lists_to_tuples(regular_tracer).layout.form
        == ak.lists_to_tuples(regular).layout.form
    )


def test_union_lists_share_width():
    layout = ak.contents.UnionArray(
        ak.index.Index8(np.array([0, 1, 0], dtype=np.int8)),
        ak.index.Index64(np.array([0, 0, 1], dtype=np.int64)),
        [ak.to_layout([[1], [2, 3]]), ak.to_layout([[4, 5, 6]])],
    )
    result = ak.lists_to_tuples(ak.Array(layout))
    assert result.to_list() == [(1, None, None), (4, 5, 6), (2, 3, None)]
    assert ak.validity_error(result) == ""


def test_randomized_tuple_conversion():
    import random

    rng = random.Random(200)
    for _ in range(30):
        values = [
            None
            if rng.randrange(6) == 0
            else tuple(
                rng.choice([None, -1, 0, 2, "x"]) for _ in range(rng.randrange(5))
            )
            for _ in range(8)
        ]
        if all(value is None for value in values):
            continue
        array = ak.Array(values)
        for drop_none in [True, False]:
            width = max(len(value) for value in values if value is not None)
            expected = [
                None
                if value is None
                else [item for item in value if item is not None]
                if drop_none
                else [*value, *([None] * (width - len(value)))]
                for value in values
            ]
            result = ak.tuples_to_lists(array, drop_none=drop_none, pad=not drop_none)
            assert result.to_list() == expected
            assert ak.validity_error(result) == ""


def test_zero_size_numpy_dimensions():
    for shape in [(0, 3), (2, 0), (2, 3, 0), (0, 3, 0)]:
        array = ak.Array(np.empty(shape, dtype=np.int64))
        result = ak.lists_to_tuples(array)
        assert ak.validity_error(result) == ""
        assert len(result) == shape[0]
        if shape[-1] == 0:
            assert result.layout.form == ak.lists_to_tuples(array, size=0).layout.form


def test_dispatch():
    class Override:
        def __awkward_function__(self, func, array_likes, args, kwargs):
            return func.__name__

    assert ak.tuples_to_lists(Override()) == "tuples_to_lists"
    assert ak.lists_to_tuples(Override()) == "lists_to_tuples"


def test_negative_named_axes():
    array = ak.with_named_axis(ak.Array([(1, 2), (3, 4)]), {"events": -1})
    result = ak.tuples_to_lists(array)
    assert result.named_axis == {"events": -2}
    assert ak.lists_to_tuples(result).named_axis == {"events": -1}

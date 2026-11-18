# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

import pytest

import awkward as ak


@pytest.mark.parametrize("typetracer", [False, True])
@pytest.mark.parametrize(
    "data",
    [
        ["ab", "c"],
        [[1, 2], [3]],
        [([1, 2], [3]), ([], [4, 5])],
        [([1, 2], 3), ([], 4)],
    ],
)
def test_tuples_to_lists_requires_a_tuple_at_axis(data, typetracer):
    array = ak.Array(data)
    if typetracer:
        array = ak.Array(array.layout.to_typetracer(forget_length=True))
    with pytest.raises(ak.errors.AxisError, match="no tuple records at axis=-1"):
        ak.tuples_to_lists(array)


def test_tuples_to_lists_selects_outer_tuple_at_shared_depth():
    array = ak.Array([((1, 2), (3, 4)), ((5, 6), (7, 8))])
    expected = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
    assert ak.tuples_to_lists(array).to_list() == expected
    assert ak.tuples_to_lists(array, axis=1).to_list() == expected
    with pytest.raises(ak.errors.AxisError, match="axis=2 exceeds the depth"):
        ak.tuples_to_lists(array, axis=2)


def test_lists_to_tuples_infers_only_referenced_union_lists():
    import numpy as np

    array = ak.Array(
        ak.contents.UnionArray(
            ak.index.Index8(np.array([0, 0, 1], dtype=np.int8)),
            ak.index.Index64(np.array([0, 0, 0], dtype=np.int64)),
            [
                ak.contents.ListOffsetArray(
                    ak.index.Index64(np.array([0, 1, 6], dtype=np.int64)),
                    ak.contents.NumpyArray(np.arange(6)),
                ),
                ak.contents.NumpyArray(np.array([1.5])),
            ],
        )
    )
    assert ak.lists_to_tuples(array, axis=1).to_list() == [(0,), (0,), 1.5]


def test_regular_conversion_preserves_nonoptional_fields():
    array = ak.to_regular([[1, 2], [3, 4]])
    result = ak.lists_to_tuples(array)
    assert str(result.type) == "2 * (int64, int64)"
    assert ak.tuples_to_lists(result).layout.is_regular
    assert ak.tuples_to_lists(result).layout.form == array.layout.form
    assert ak.tuples_to_lists(result).to_list() == array.to_list()
    tracer = ak.Array(array.layout.to_typetracer(forget_length=True))
    assert ak.lists_to_tuples(tracer).layout.form == result.layout.form
    assert (
        ak.tuples_to_lists(ak.lists_to_tuples(tracer)).layout.form == array.layout.form
    )
    assert ak.lists_to_tuples(array, size=1).to_list() == [(1,), (3,)]
    assert ak.lists_to_tuples(array, size=3).to_list() == [
        (1, 2, None),
        (3, 4, None),
    ]


def test_inferred_width_preserves_unrelated_buffers():
    import numpy as np

    values = np.arange(3)
    array = ak.Array(
        ak.contents.RecordArray(
            [ak.to_layout([[1], [2, 3], []]), ak.contents.NumpyArray(values)],
            ["items", "values"],
        )
    )
    result = ak.lists_to_tuples(array)
    assert np.shares_memory(result.layout.content("values").data, values)
    assert result.to_list() == [
        {"items": (1, None), "values": 0},
        {"items": (2, 3), "values": 1},
        {"items": (None, None), "values": 2},
    ]


@pytest.mark.parametrize("axis", [1, -1])
def test_conversion_defaults_preserve_positions_and_roundtrip_axis(axis):
    array = ak.Array([(1, None, 3), (4, 5, 6)])
    lists = ak.tuples_to_lists(array, axis=axis)
    assert lists.to_list() == [[1, None, 3], [4, 5, 6]]
    assert ak.lists_to_tuples(lists, axis=axis).to_list() == array.to_list()


def test_conversion_preserves_variant_widths_by_default():
    array = ak.Array([(1,), (2, None, 3)])
    assert ak.tuples_to_lists(array).to_list() == [[1], [2, None, 3]]
    assert ak.tuples_to_lists(array, pad=True).to_list() == [
        [1, None, None],
        [2, None, 3],
    ]
    assert ak.tuples_to_lists(array, drop_none=True).to_list() == [[1], [2, 3]]


def test_boolean_numeric_fields_promote():
    result = ak.tuples_to_lists([(True, 2), (False, 3)])
    assert result.to_list() == [[1, 2], [0, 3]]
    assert str(result.type) == "2 * 2 * int64"


@pytest.mark.parametrize("operation", ["tuples_to_lists", "lists_to_tuples"])
@pytest.mark.parametrize("mask_kind", ["byte", "bit", "unmasked"])
def test_conversion_preserves_mask_wrappers(operation, mask_kind):
    import numpy as np

    convert = getattr(ak, operation)
    content = (
        ak.to_layout([(1, 2), (3, 4)])
        if operation == "tuples_to_lists"
        else ak.to_layout([[1, 2], [3, 4]])
    )
    if mask_kind == "byte":
        layout = ak.contents.ByteMaskedArray(
            ak.index.Index8(np.array([1, 0], dtype=np.int8)), content, True
        )
    elif mask_kind == "bit":
        layout = ak.contents.BitMaskedArray(
            ak.index.IndexU8(np.array([1], dtype=np.uint8)), content, True, 2, True
        )
    else:
        layout = ak.contents.UnmaskedArray(content)
    kwargs = {} if operation == "tuples_to_lists" else {"size": 2}
    result = convert(ak.Array(layout), **kwargs)
    assert type(result.layout) is type(layout)
    expected = [[1, 2], [3, 4]] if operation == "tuples_to_lists" else [(1, 2), (3, 4)]
    if mask_kind != "unmasked":
        expected[1] = None
    assert result.to_list() == expected
    tracer = ak.Array(layout.to_typetracer(forget_length=True))
    assert convert(tracer, **kwargs).layout.form == result.layout.form


@pytest.mark.parametrize("width", [0, 1, 3, 7])
def test_direct_field_carries_respect_regular_stride_and_slices(width):
    import numpy as np

    source = ak.Array(np.arange(30).reshape(5, 6))[1:4]
    result = ak.lists_to_tuples(source, size=width)
    expected = [
        (*range(start, start + min(width, 6)), *([None] * max(0, width - 6)))
        for start in (6, 12, 18)
    ]
    assert result.to_list() == expected
    tracer = ak.Array(source.layout.to_typetracer(forget_length=True))
    assert ak.lists_to_tuples(tracer, size=width).layout.form == result.layout.form


@pytest.mark.parametrize("values", [["a", None, "b"], [{"x": 1}, None, {"x": 2}]])
def test_direct_field_carries_preserve_nested_values(values):
    source = ak.Array([values, [], list(reversed(values))])
    assert ak.lists_to_tuples(source, size=4).to_list() == [
        (*values, None),
        (None, None, None, None),
        (*reversed(values), None),
    ]

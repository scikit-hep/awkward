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
def test_stack_requires_a_tuple_at_axis(data, typetracer):
    array = ak.Array(data)
    if typetracer:
        array = ak.Array(array.layout.to_typetracer(forget_length=True))
    with pytest.raises(ak.errors.AxisError, match="no tuple records at axis=-1"):
        ak.stack(array)


def test_stack_selects_outer_tuple_at_shared_depth():
    array = ak.Array([((1, 2), (3, 4)), ((5, 6), (7, 8))])
    expected = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
    assert ak.stack(array).to_list() == expected
    assert ak.stack(array, axis=0).to_list() == expected
    with pytest.raises(ak.errors.AxisError, match="axis=1 exceeds the depth"):
        ak.stack(array, axis=1)


def test_unstack_infers_only_referenced_union_lists():
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
    assert ak.unstack(array, axis=1).to_list() == [(0,), (0,), 1.5]


def test_regular_conversion_preserves_nonoptional_fields():
    array = ak.to_regular([[1, 2], [3, 4]])
    result = ak.unstack(array)
    assert str(result.type) == "2 * (int64, int64)"
    assert ak.stack(result).layout.is_regular
    assert ak.stack(result).layout.form == array.layout.form
    assert ak.stack(result).to_list() == array.to_list()
    tracer = ak.Array(array.layout.to_typetracer(forget_length=True))
    assert ak.unstack(tracer).layout.form == result.layout.form
    assert ak.stack(ak.unstack(tracer)).layout.form == array.layout.form
    assert ak.unstack(array, size=1).to_list() == [(1,), (3,)]
    assert ak.unstack(array, size=3).to_list() == [
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
    result = ak.unstack(array)
    assert np.shares_memory(result.layout.content("values").data, values)
    assert result.to_list() == [
        {"items": (1, None), "values": 0},
        {"items": (2, 3), "values": 1},
        {"items": (None, None), "values": 2},
    ]

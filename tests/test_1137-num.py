# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest

import awkward as ak

to_list = ak.operations.to_list


def test_bytemaskedarray_num():
    content = ak.operations.from_iter(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, 8.8, 9.9]],
            [[], [10.0, 11.1, 12.2]],
        ],
        highlevel=False,
    )
    mask = ak.index.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.contents.ByteMaskedArray(mask, content, valid_when=False)

    assert to_list(array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.num(array, axis=0) == 5
    assert ak.num(array, axis=-3) == 5
    assert to_list(ak.num(array, axis=1)) == [3, 0, None, None, 2]
    assert to_list(ak.num(array, axis=-2)) == [3, 0, None, None, 2]
    assert to_list(ak.num(array, axis=2)) == [[3, 0, 2], [], None, None, [0, 3]]
    assert to_list(ak.num(array, axis=-1)) == [[3, 0, 2], [], None, None, [0, 3]]


def test_emptyarray():
    array = ak.contents.EmptyArray()
    assert to_list(ak.num(array, 0)) == 0
    assert to_list(ak.num(array, -1)) == 0
    with pytest.raises(np.AxisError) as err:
        ak.num(array, 1)
    assert "axis=1 exceeds the depth" in str(err.value)


def test_numpyarray():
    array = ak.contents.NumpyArray(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7))

    assert ak.num(array, 0) == 2
    assert to_list(ak.num(array, 1)) == [3, 3]
    assert to_list(ak.num(array, axis=2)) == [[5, 5, 5], [5, 5, 5]]
    assert to_list(ak.num(array, 3)) == [
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
    ]
    with pytest.raises(np.AxisError) as err:
        ak.num(array, 4)
    assert "axis=4 exceeds the depth" in str(err.value)


def test_regulararray():
    array = ak.contents.NumpyArray(
        np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    ).to_RegularArray()

    assert ak.num(array, 0) == 2
    assert to_list(ak.num(array, 1)) == [3, 3]
    assert to_list(ak.num(array, 2)) == [[5, 5, 5], [5, 5, 5]]
    assert to_list(ak.num(array, 3)) == [
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
    ]
    with pytest.raises(np.AxisError) as err:
        ak.num(array, 4)
    assert "axis=4 exceeds the depth" in str(err.value)

    empty = ak.contents.RegularArray(
        ak.highlevel.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
    )

    assert ak.num(empty, axis=0) == 0
    assert to_list(ak.num(empty, axis=1)) == []
    assert to_list(ak.num(empty, axis=2)) == []


def test_listarray():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    starts = ak.index.Index64(np.array([0, 3, 3], dtype=np.int64))
    stops = ak.index.Index64(np.array([3, 3, 5], dtype=np.int64))
    array = ak.contents.ListArray(starts, stops, content)

    assert to_list(array) == [
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
        [],
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
    ]

    assert to_list(ak.num(array, 0)) == 3
    assert to_list(ak.num(array, 1)) == [3, 0, 2]
    assert to_list(ak.num(array, 2)) == [[3, 3, 3], [], [3, 3]]
    assert to_list(ak.num(array, 3)) == [
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [],
        [[2, 2, 2], [2, 2, 2]],
    ]
    with pytest.raises(np.AxisError) as err:
        ak.num(array, 4)
    assert "axis=4 exceeds the depth" in str(err.value)


def test_listoffsetarray():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    offsets = ak.index.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    array = ak.contents.ListOffsetArray(offsets, content)

    assert to_list(array) == [
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
        [],
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
    ]

    assert to_list(ak.num(array, 0)) == 3
    assert to_list(ak.num(array, 1)) == [3, 0, 2]
    assert to_list(ak.num(array, 2)) == [[3, 3, 3], [], [3, 3]]
    assert to_list(ak.num(array, 3)) == [
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [],
        [[2, 2, 2], [2, 2, 2]],
    ]
    with pytest.raises(np.AxisError) as err:
        ak.num(array, 4)
    assert "axis=4 exceeds the depth" in str(err.value)


def test_indexedarray():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    offsets = ak.index.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listarray = ak.contents.ListOffsetArray(offsets, content)
    index = ak.index.Index64(np.array([2, 2, 1, 0], dtype=np.int64))
    array = ak.contents.IndexedArray(index, listarray)

    assert to_list(array) == [
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
        [],
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
    ]

    assert to_list(ak.num(array, 0)) == 4
    assert to_list(ak.num(array, 1)) == [2, 2, 0, 3]
    assert to_list(ak.num(array, 2)) == [[3, 3], [3, 3], [], [3, 3, 3]]
    assert to_list(ak.num(array, 3)) == [
        [[2, 2, 2], [2, 2, 2]],
        [[2, 2, 2], [2, 2, 2]],
        [],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    ]

    with pytest.raises(np.AxisError) as err:
        ak.num(array, 4)
    assert "axis=4 exceeds the depth" in str(err.value)


def test_indexedoptionarray():
    content = ak.contents.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    offsets = ak.index.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listarray = ak.contents.ListOffsetArray(offsets, content)
    index = ak.index.Index64(np.array([2, -1, 2, 1, -1, 0], dtype=np.int64))
    array = ak.contents.IndexedOptionArray(index, listarray)

    assert to_list(array) == [
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
        None,
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
        [],
        None,
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
    ]

    assert to_list(ak.num(array, 0)) == 6
    assert to_list(ak.num(array, 1)) == [2, None, 2, 0, None, 3]
    assert to_list(ak.num(array, 2)) == [[3, 3], None, [3, 3], [], None, [3, 3, 3]]
    assert to_list(ak.num(array, 3)) == [
        [[2, 2, 2], [2, 2, 2]],
        None,
        [[2, 2, 2], [2, 2, 2]],
        [],
        None,
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    ]

    with pytest.raises(np.AxisError) as err:
        ak.num(array, 4)
    assert "axis=4 exceeds the depth" in str(err.value)


def test_recordarray():
    array = ak.operations.from_iter(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
        ],
        highlevel=False,
    )

    assert ak.num(array, 0) == 4

    array = ak.operations.from_iter(
        [
            {"x": [3.3, 3.3, 3.3], "y": []},
            {"x": [2.2, 2.2], "y": [1]},
            {"x": [1.1], "y": [2, 2]},
            {"x": [], "y": [3, 3, 3]},
        ],
        highlevel=False,
    )

    assert ak.num(array, 0) == 4
    assert to_list(ak.num(array, 1)) == [
        {"x": 3, "y": 0},
        {"x": 2, "y": 1},
        {"x": 1, "y": 2},
        {"x": 0, "y": 3},
    ]
    assert to_list(ak.num(array, 1)[2]) == {"x": 1, "y": 2}

    array = ak.operations.from_iter(
        [
            {"x": [[3.3, 3.3, 3.3]], "y": []},
            {"x": [[2.2, 2.2]], "y": [1]},
            {"x": [[1.1]], "y": [2, 2]},
            {"x": [[]], "y": [3, 3, 3]},
        ],
        highlevel=False,
    )

    assert ak.num(array, 0) == 4
    assert to_list(ak.num(array, 1)) == [
        {"x": 1, "y": 0},
        {"x": 1, "y": 1},
        {"x": 1, "y": 2},
        {"x": 1, "y": 3},
    ]
    assert to_list(ak.num(array, 1)[2]) == {"x": 1, "y": 2}


def test_unionarray():
    content1 = ak.operations.from_iter(
        [[], ["1"], ["2", "2"], ["3", "3", "3"]], highlevel=False
    )
    content2 = ak.operations.from_iter(
        [[3.3, 3.3, 3.3], [2.2, 2.2], [1.1], []], highlevel=False
    )
    tags = ak.index.Index8(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.index.Index64(np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64))
    array = ak.contents.UnionArray(tags, index, [content1, content2])
    assert to_list(array) == [
        [],
        [3.3, 3.3, 3.3],
        ["1"],
        [2.2, 2.2],
        ["2", "2"],
        [1.1],
        ["3", "3", "3"],
        [],
    ]

    assert ak.num(array, 0) == 8
    assert isinstance(ak.num(array, 1, highlevel=False), ak.contents.NumpyArray)
    assert to_list(ak.num(array, 1)) == [0, 3, 1, 2, 2, 1, 3, 0]


def test_highlevel():
    array = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert to_list(ak.operations.num(array)) == [3, 0, 2]


def test_array_3d():
    array = ak.highlevel.Array(np.arange(3 * 5 * 2).reshape(3, 5, 2)).layout

    assert to_list(array) == [
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
        [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]],
    ]
    assert ak.num(array, axis=0) == 3
    assert to_list(ak.num(array, axis=1)) == [5, 5, 5]
    assert to_list(ak.num(array, axis=2)) == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ]
    with pytest.raises(ValueError) as err:
        assert ak.num(array, axis=3)
        assert str(err.value).startswith("axis=3 exceeds the depth of this array")

    assert to_list(ak.num(array, axis=-1)) == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ]
    assert to_list(ak.num(array, axis=-2)) == [5, 5, 5]
    assert ak.num(array, axis=-3) == 3

    with pytest.raises(ValueError) as err:
        assert ak.num(array, axis=-4)
        assert str(err.value).startswith("axis=-4 exceeds the depth of this array")

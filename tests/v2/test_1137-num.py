# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)


def test_bytemaskedarray_num():
    content = ak.from_iter(
        [
            [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
            [],
            [[5.5]],
            [[6.6, 7.7, 8.8, 9.9]],
            [[], [10.0, 11.1, 12.2]],
        ],
        highlevel=False,
    )
    mask = ak.layout.Index8(np.array([0, 0, 1, 1, 0], dtype=np.int8))
    array = ak.layout.ByteMaskedArray(mask, content, valid_when=False)

    array = v1_to_v2(array)

    assert ak.to_list(array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert array.num(axis=0) == 5
    assert array.num(axis=-3) == 5
    assert ak.to_list(array.num(axis=1)) == [3, 0, None, None, 2]
    assert ak.to_list(array.num(axis=-2)) == [3, 0, None, None, 2]
    assert ak.to_list(array.num(axis=2)) == [[3, 0, 2], [], None, None, [0, 3]]
    assert ak.to_list(array.num(axis=-1)) == [[3, 0, 2], [], None, None, [0, 3]]


def test_emptyarray():
    array = ak.layout.EmptyArray()
    array = v1_to_v2(array)
    assert ak.to_list(array.num(0)) == 0
    assert ak.to_list(array.num(1)) == []
    assert ak.to_list(array.num(2)) == []


def test_numpyarray():
    array = ak.layout.NumpyArray(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7))
    array = v1_to_v2(array)

    assert array.num(0) == 2
    assert ak.to_list(array.num(1)) == [3, 3]
    assert ak.to_list(array.num(axis=2)) == [[5, 5, 5], [5, 5, 5]]
    assert ak.to_list(array.num(3)) == [
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
    ]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("axis=4 exceeds the depth of this array (3)")


def test_regulararray():
    array = ak.layout.NumpyArray(
        np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    ).toRegularArray()

    array = v1_to_v2(array)

    assert array.num(0) == 2
    assert ak.to_list(array.num(1)) == [3, 3]
    assert ak.to_list(array.num(2)) == [[5, 5, 5], [5, 5, 5]]
    assert ak.to_list(array.num(3)) == [
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
        [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 7, 7, 7, 7]],
    ]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("axis=4 exceeds the depth of this array (3)")

    empty = ak.Array(
        ak.layout.RegularArray(
            ak.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
        )
    )

    empty = v1_to_v2(empty.layout)

    assert empty.num(axis=0) == 0
    assert ak.to_list(empty.num(axis=1)) == []
    assert ak.to_list(empty.num(axis=2)) == []


def test_listarray():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    starts = ak.layout.Index64(np.array([0, 3, 3], dtype=np.int64))
    stops = ak.layout.Index64(np.array([3, 3, 5], dtype=np.int64))
    array = ak.layout.ListArray64(starts, stops, content)

    array = v1_to_v2(array)

    assert ak.to_list(array) == [
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
        [],
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
    ]

    assert ak.to_list(array.num(0)) == 3
    assert ak.to_list(array.num(1)) == [3, 0, 2]
    assert ak.to_list(array.num(2)) == [[3, 3, 3], [], [3, 3]]
    assert ak.to_list(array.num(3)) == [
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [],
        [[2, 2, 2], [2, 2, 2]],
    ]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("axis=4 exceeds the depth of this array (3)")


def test_listoffsetarray():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    array = ak.layout.ListOffsetArray64(offsets, content)

    array = v1_to_v2(array)

    assert ak.to_list(array) == [
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
        [],
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
    ]

    assert ak.to_list(array.num(0)) == 3
    assert ak.to_list(array.num(1)) == [3, 0, 2]
    assert ak.to_list(array.num(2)) == [[3, 3, 3], [], [3, 3]]
    assert ak.to_list(array.num(3)) == [
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [],
        [[2, 2, 2], [2, 2, 2]],
    ]
    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("axis=4 exceeds the depth of this array (3)")


def test_indexedarray():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listarray = ak.layout.ListOffsetArray64(offsets, content)
    index = ak.layout.Index64(np.array([2, 2, 1, 0], dtype=np.int64))
    array = ak.layout.IndexedArray64(index, listarray)

    array = v1_to_v2(array)

    assert ak.to_list(array) == [
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
        [[[18, 19], [20, 21], [22, 23]], [[24, 25], [26, 27], [28, 29]]],
        [],
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
            [[12, 13], [14, 15], [16, 17]],
        ],
    ]

    assert ak.to_list(array.num(0)) == 4
    assert ak.to_list(array.num(1)) == [2, 2, 0, 3]
    assert ak.to_list(array.num(2)) == [[3, 3], [3, 3], [], [3, 3, 3]]
    assert ak.to_list(array.num(3)) == [
        [[2, 2, 2], [2, 2, 2]],
        [[2, 2, 2], [2, 2, 2]],
        [],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    ]

    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("axis=4 exceeds the depth of this array (3)")


def test_indexedoptionarray():
    content = ak.layout.NumpyArray(np.arange(2 * 3 * 5).reshape(5, 3, 2))
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5], dtype=np.int64))
    listarray = ak.layout.ListOffsetArray64(offsets, content)
    index = ak.layout.Index64(np.array([2, -1, 2, 1, -1, 0], dtype=np.int64))
    array = ak.layout.IndexedOptionArray64(index, listarray)

    array = v1_to_v2(array)

    assert ak.to_list(array) == [
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

    assert ak.to_list(array.num(0)) == 6
    assert ak.to_list(array.num(1)) == [2, None, 2, 0, None, 3]
    assert ak.to_list(array.num(2)) == [[3, 3], None, [3, 3], [], None, [3, 3, 3]]
    assert ak.to_list(array.num(3)) == [
        [[2, 2, 2], [2, 2, 2]],
        None,
        [[2, 2, 2], [2, 2, 2]],
        [],
        None,
        [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
    ]

    with pytest.raises(ValueError) as err:
        array.num(4)
    assert str(err.value).startswith("axis=4 exceeds the depth of this array (3)")


def test_recordarray():
    array = ak.from_iter(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
        ],
        highlevel=False,
    )

    array = v1_to_v2(array)

    assert ak.to_list(array.num(0)) == {"x": 4, "y": 4}

    array = ak.from_iter(
        [
            {"x": [3.3, 3.3, 3.3], "y": []},
            {"x": [2.2, 2.2], "y": [1]},
            {"x": [1.1], "y": [2, 2]},
            {"x": [], "y": [3, 3, 3]},
        ],
        highlevel=False,
    )

    array = v1_to_v2(array)

    assert ak.to_list(array.num(0)) == {"x": 4, "y": 4}
    assert ak.to_list(array.num(1)) == [
        {"x": 3, "y": 0},
        {"x": 2, "y": 1},
        {"x": 1, "y": 2},
        {"x": 0, "y": 3},
    ]
    assert ak.to_list(array.num(1)[2]) == {"x": 1, "y": 2}

    array = ak.from_iter(
        [
            {"x": [[3.3, 3.3, 3.3]], "y": []},
            {"x": [[2.2, 2.2]], "y": [1]},
            {"x": [[1.1]], "y": [2, 2]},
            {"x": [[]], "y": [3, 3, 3]},
        ],
        highlevel=False,
    )

    array = v1_to_v2(array)

    assert ak.to_list(array.num(0)) == {"x": 4, "y": 4}
    assert ak.to_list(array.num(1)) == [
        {"x": 1, "y": 0},
        {"x": 1, "y": 1},
        {"x": 1, "y": 2},
        {"x": 1, "y": 3},
    ]
    assert ak.to_list(array.num(1)[2]) == {"x": 1, "y": 2}


def test_unionarray():
    content1 = ak.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    content2 = ak.from_iter([[3.3, 3.3, 3.3], [2.2, 2.2], [1.1], []], highlevel=False)
    tags = ak.layout.Index8(np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64))
    array = ak.layout.UnionArray8_64(tags, index, [content1, content2])
    assert ak.to_list(array) == [
        [],
        [3.3, 3.3, 3.3],
        [1],
        [2.2, 2.2],
        [2, 2],
        [1.1],
        [3, 3, 3],
        [],
    ]

    assert array.num(0) == 8
    assert isinstance(array.num(1), ak.layout.NumpyArray)
    assert ak.to_list(array.num(1)) == [0, 3, 1, 2, 2, 1, 3, 0]


def test_highlevel():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    assert ak.to_list(ak.num(array)) == [3, 0, 2]


def test_array_3d():
    array = ak.Array(np.arange(3 * 5 * 2).reshape(3, 5, 2))

    array = v1_to_v2(array.layout)

    assert ak.to_list(array) == [
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
        [[20, 21], [22, 23], [24, 25], [26, 27], [28, 29]],
    ]
    assert array.num(axis=0) == 3
    assert ak.to_list(array.num(axis=1)) == [5, 5, 5]
    assert ak.to_list(array.num(axis=2)) == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ]
    with pytest.raises(ValueError) as err:
        assert array.num(axis=3)
        assert str(err.value).startswith("axis=3 exceeds the depth of this array (2)")

    assert ak.to_list(array.num(axis=-1)) == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ]
    assert ak.to_list(array.num(axis=-2)) == [5, 5, 5]
    assert array.num(axis=-3) == 3

    with pytest.raises(ValueError) as err:
        assert array.num(axis=-4)
        assert str(err.value).startswith("axis=-4 exceeds the depth of this array (3)")


def test_listarray_negative_axis_wrap():
    array = ak.Array(np.arange(3 * 5 * 2).reshape(3, 5, 2).tolist())
    assert ak.num(array, axis=0) == 3
    assert ak.num(array, axis=1).tolist() == [5, 5, 5]
    assert ak.num(array, axis=2).tolist() == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ]

    with pytest.raises(ValueError) as err:
        assert ak.num(array, axis=3)
        assert str(err.value).startswith("axis=3 exceeds the depth of this array (2)")

    assert ak.num(array, axis=-1).tolist() == [
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2],
    ]
    assert ak.num(array, axis=-2).tolist() == [5, 5, 5]
    assert ak.num(array, axis=-3) == 3
    with pytest.raises(ValueError) as err:
        assert ak.num(array, axis=-4)
        assert str(err.value).startswith("axis=-4 exceeds the depth of this array (3)")


def test_recordarray_negative_axis_wrap():
    array = ak.Array(
        [
            {"x": [1], "y": [[], [1]]},
            {"x": [1, 2], "y": [[], [1], [1, 2]]},
            {"x": [1, 2, 3], "y": [[], [1], [1, 2], [1, 2, 3]]},
        ]
    )

    array = v1_to_v2(array.layout)

    assert ak.to_list(array.num(axis=0)) == {"x": 3, "y": 3}
    assert ak.to_list(array.num(axis=1)) == [
        {"x": 1, "y": 2},
        {"x": 2, "y": 3},
        {"x": 3, "y": 4},
    ]
    with pytest.raises(ValueError) as err:
        assert array.num(axis=2)
    assert str(err.value).startswith("axis=2 exceeds the depth of this array (1)")

    assert ak.to_list(array.num(axis=-1)) == [
        {"x": 1, "y": [0, 1]},
        {"x": 2, "y": [0, 1, 2]},
        {"x": 3, "y": [0, 1, 2, 3]},
    ]


def test_recordarray_axis_out_of_range():
    array = ak.Array(
        [
            {"x": [1], "y": [[], [1]]},
            {"x": [1, 2], "y": [[], [1], [1, 2]]},
            {"x": [1, 2, 3], "y": [[], [1], [1, 2], [1, 2, 3]]},
        ]
    )

    array = v1_to_v2(array.layout)

    with pytest.raises(ValueError) as err:
        assert array.num(axis=-2)
        assert str(err.value).startswith("axis=-2 exceeds the depth of this array (2)")

    with pytest.raises(ValueError) as err:
        assert array.num(axis=-3)
        assert str(err.value).startswith("axis=-2 exceeds the depth of this array (2)")

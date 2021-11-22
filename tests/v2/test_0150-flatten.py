# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401

from awkward._v2.tmp_for_testing import v1_to_v2

pytestmark = pytest.mark.skipif(
    ak._util.py27, reason="No Python 2.7 support in Awkward 2.x"
)

# includes test_0117, test_0110, test_0042, test_0127,
# test_0198, test_0446, test_0585, test_0590


def test_flatten_ListOffsetArray():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    array = v1_to_v2(array.layout)

    assert ak.to_list(array.flatten()) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert ak.to_list(array[1:].flatten()) == [4.4, 5.5]

    array = ak.Array(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]]
    )
    array = v1_to_v2(array.layout)

    assert ak.to_list(array.flatten()) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array[1:].flatten()) == [[5.5], [], [6.6, 7.7, 8.8, 9.9]]
    assert ak.to_list(array[:, 1:].flatten()) == [
        [],
        [3.3, 4.4],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array.flatten(axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array[1:].flatten(axis=2)) == [
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array[:, 1:].flatten(axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.Array(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).tolist())
    array = v1_to_v2(array.layout)

    assert (
        ak.to_list(array.flatten(axis=1))
        == np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.to_list(array.flatten(axis=2))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.to_list(array.flatten(axis=3))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )

    def toListArray(x):
        if isinstance(x, ak.layout.ListOffsetArray64):
            starts = ak.layout.Index64(np.asarray(x.offsets)[:-1])
            stops = ak.layout.Index64(np.asarray(x.offsets)[1:])
            return ak.layout.ListArray64(starts, stops, toListArray(x.content))
        elif isinstance(x, ak.layout.ListOffsetArray32):
            starts = ak.layout.Index64(np.asarray(x.offsets)[:-1])
            stops = ak.layout.Index64(np.asarray(x.offsets)[1:])
            return ak.layout.ListArray32(starts, stops, toListArray(x.content))
        elif isinstance(x, ak.layout.ListOffsetArrayU32):
            starts = ak.layout.Index64(np.asarray(x.offsets)[:-1])
            stops = ak.layout.Index64(np.asarray(x.offsets)[1:])
            return ak.layout.ListArrayU32(starts, stops, toListArray(x.content))
        else:
            return x

    array = ak.Array(
        toListArray(
            ak.from_iter(
                np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7).tolist(), highlevel=False
            )
        )
    )
    array = v1_to_v2(array.layout)

    assert (
        ak.to_list(array.flatten(axis=1))
        == np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.to_list(array.flatten(axis=2))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.to_list(array.flatten(axis=3))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )

    array = ak.Array(np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7))
    array = v1_to_v2(array.layout)

    assert (
        ak.to_list(array.flatten(axis=1))
        == np.arange(2 * 3 * 5 * 7).reshape(2 * 3, 5, 7).tolist()
    )
    assert (
        ak.to_list(array.flatten(axis=2))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3 * 5, 7).tolist()
    )
    assert (
        ak.to_list(array.flatten(axis=3))
        == np.arange(2 * 3 * 5 * 7).reshape(2, 3, 5 * 7).tolist()
    )


def test_flatten_IndexedArray():
    array = ak.Array([[1.1, 2.2, None, 3.3], None, [], None, [4.4, 5.5], None])
    array = v1_to_v2(array.layout)

    assert ak.to_list(array.flatten()) == [1.1, 2.2, None, 3.3, 4.4, 5.5]
    assert ak.to_list(array[1:].flatten()) == [4.4, 5.5]

    array = ak.Array(
        [
            [[0.0, 1.1, 2.2], None, None, [3.3, 4.4]],
            [],
            [[5.5]],
            [[], [6.6, 7.7, 8.8, 9.9]],
        ]
    )
    array = v1_to_v2(array.layout)

    assert ak.to_list(array.flatten(axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array[1:].flatten(axis=2)) == [
        [],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array[:, 1:].flatten(axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.Array(
        [
            [[0.0, 1.1, 2.2], [3.3, 4.4]],
            [],
            [[5.5]],
            None,
            None,
            [[], [6.6, 7.7, 8.8, 9.9]],
        ]
    )
    array = v1_to_v2(array.layout)

    assert ak.to_list(array.flatten(axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        [5.5],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array[1:].flatten(axis=2)) == [
        [],
        [5.5],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array[:, 1:].flatten(axis=2)) == [
        [3.3, 4.4],
        [],
        [],
        None,
        None,
        [6.6, 7.7, 8.8, 9.9],
    ]

    array = ak.Array(
        [
            [[0.0, 1.1, None, 2.2], None, [], None, [3.3, 4.4]],
            None,
            [],
            [[5.5]],
            None,
            [[], [6.6, None, 7.7, 8.8, 9.9], None],
        ]
    )
    array = v1_to_v2(array.layout)

    assert ak.to_list(array.flatten()) == [
        [0.0, 1.1, None, 2.2],
        None,
        [],
        None,
        [3.3, 4.4],
        [5.5],
        [],
        [6.6, None, 7.7, 8.8, 9.9],
        None,
    ]
    assert ak.to_list(array.flatten(axis=2)) == [
        [0.0, 1.1, None, 2.2, 3.3, 4.4],
        None,
        [],
        [5.5],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array[1:].flatten(axis=2)) == [
        None,
        [],
        [5.5],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array[:, 1:].flatten(axis=2)) == [
        [3.3, 4.4],
        None,
        [],
        [],
        None,
        [6.6, None, 7.7, 8.8, 9.9],
    ]

    content = ak.from_iter(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]], highlevel=False
    )
    index = ak.layout.Index64(np.array([2, 1, 0, 3, 3, 4], dtype=np.int64))
    array = ak.Array(ak.layout.IndexedArray64(index, content))
    array = v1_to_v2(array.layout)

    assert ak.to_list(array) == [
        [3.3, 4.4],
        [],
        [0.0, 1.1, 2.2],
        [5.5],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array.flatten()) == [
        3.3,
        4.4,
        0.0,
        1.1,
        2.2,
        5.5,
        5.5,
        6.6,
        7.7,
        8.8,
        9.9,
    ]

    content = ak.from_iter(
        [[[0.0, 1.1, 2.2], [], [3.3, 4.4]], [], [[5.5]], [[], [6.6, 7.7, 8.8, 9.9]]],
        highlevel=False,
    )
    index = ak.layout.Index64(np.array([2, 2, 1, 0, 3], dtype=np.int64))
    array = ak.Array(ak.layout.IndexedArray64(index, content))
    array = v1_to_v2(array.layout)

    assert ak.to_list(array) == [
        [[5.5]],
        [[5.5]],
        [],
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [[], [6.6, 7.7, 8.8, 9.9]],
    ]
    assert ak.to_list(array.flatten(axis=2)) == [
        [5.5],
        [5.5],
        [],
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [6.6, 7.7, 8.8, 9.9],
    ]


def test_flatten_RecordArray():
    array = ak.Array(
        [
            {"x": [], "y": [[3, 3, 3]]},
            {"x": [[1]], "y": [[2, 2]]},
            {"x": [[2], [2]], "y": [[1]]},
            {"x": [[3], [3], [3]], "y": [[]]},
        ]
    )
    array = v1_to_v2(array.layout)

    assert ak.to_list(array.flatten(axis=2)) == [
        {"x": [], "y": [3, 3, 3]},
        {"x": [1], "y": [2, 2]},
        {"x": [2, 2], "y": [1]},
        {"x": [3, 3, 3], "y": []},
    ]
    assert ak.to_list(array[1:].flatten(axis=2)) == [
        {"x": [1], "y": [2, 2]},
        {"x": [2, 2], "y": [1]},
        {"x": [3, 3, 3], "y": []},
    ]
    assert ak.to_list(array[:, 1:].flatten(axis=2)) == [
        {"x": [], "y": []},
        {"x": [], "y": []},
        {"x": [2], "y": []},
        {"x": [3, 3], "y": []},
    ]


def test_flatten_UnionArray():
    content1 = ak.from_iter([[1.1], [2.2, 2.2], [3.3, 3.3, 3.3]], highlevel=False)
    content2 = ak.from_iter(
        [[[3, 3, 3], [3, 3, 3], [3, 3, 3]], [[2, 2], [2, 2]], [[1]]], highlevel=False
    )
    tags = ak.layout.Index8(np.array([0, 1, 0, 1, 0, 1], dtype=np.int8))
    index = ak.layout.Index64(np.array([0, 0, 1, 1, 2, 2], dtype=np.int64))
    array = ak.Array(ak.layout.UnionArray8_64(tags, index, [content1, content2]))
    array = v1_to_v2(array.layout)

    assert ak.to_list(array) == [
        [1.1],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [2.2, 2.2],
        [[2, 2], [2, 2]],
        [3.3, 3.3, 3.3],
        [[1]],
    ]
    assert ak.to_list(array[1:]) == [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [2.2, 2.2],
        [[2, 2], [2, 2]],
        [3.3, 3.3, 3.3],
        [[1]],
    ]
    assert ak.to_list(array.flatten()) == [
        1.1,
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        2.2,
        2.2,
        [2, 2],
        [2, 2],
        3.3,
        3.3,
        3.3,
        [1],
    ]
    assert ak.to_list(array[1:].flatten()) == [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        2.2,
        2.2,
        [2, 2],
        [2, 2],
        3.3,
        3.3,
        3.3,
        [1],
    ]

    array = ak.Array(ak.layout.UnionArray8_64(tags, index, [content2, content2]))
    array = v1_to_v2(array.layout)

    assert ak.to_list(array) == [
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [[2, 2], [2, 2]],
        [[2, 2], [2, 2]],
        [[1]],
        [[1]],
    ]
    assert ak.to_list(array.flatten(axis=2)) == [
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [1],
        [1],
    ]
    assert ak.to_list(array[1:].flatten(axis=2)) == [
        [3, 3, 3, 3, 3, 3, 3, 3, 3],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [1],
        [1],
    ]
    assert ak.to_list(array[:, 1:].flatten(axis=2)) == [
        [3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3],
        [2, 2],
        [2, 2],
        [],
        [],
    ]


def test_flatten():
    array = ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    array = v1_to_v2(array.layout)
    assert ak.to_list(array.flatten(axis=1)) == [1.1, 2.2, 3.3, 4.4, 5.5]


def test_flatten2():
    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )
    offsets = ak.layout.Index64(np.array([0, 3, 3, 5, 6, 10], dtype=np.int64))
    array = ak.layout.ListOffsetArray64(offsets, content)
    array = v1_to_v2(array)

    assert ak.to_list(array) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]
    assert ak.to_list(array.flatten(axis=1)) == [
        0.0,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        7.7,
        8.8,
        9.9,
    ]
    assert ak.to_list(array.flatten(axis=-1)) == [
        0.0,
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        7.7,
        8.8,
        9.9,
    ]
    with pytest.raises(ValueError) as err:
        assert ak.to_list(array.flatten(axis=-2))
        assert str(err.value).startswith("axis=0 not allowed for flatten")

    array2 = array[2:-1]
    assert ak.to_list(array2.flatten(axis=1)) == [3.3, 4.4, 5.5]
    assert ak.to_list(array2.flatten(axis=-1)) == [3.3, 4.4, 5.5]


def test_ByteMaskedArray_flatten():
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
    array = ak.Array(ak.layout.ByteMaskedArray(mask, content, valid_when=False))
    array = v1_to_v2(array.layout)

    assert ak.to_list(array) == [
        [[0.0, 1.1, 2.2], [], [3.3, 4.4]],
        [],
        None,
        None,
        [[], [10.0, 11.1, 12.2]],
    ]
    assert ak.to_list(array.flatten(axis=1)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert ak.to_list(array.flatten(axis=-2)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [],
        [10.0, 11.1, 12.2],
    ]
    assert ak.to_list(array.flatten(axis=2)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        None,
        None,
        [10.0, 11.1, 12.2],
    ]
    assert ak.to_list(array.flatten(axis=-1)) == [
        [0.0, 1.1, 2.2, 3.3, 4.4],
        [],
        None,
        None,
        [10.0, 11.1, 12.2],
    ]


@pytest.mark.skip(reason="ak.flatten() not implemented yet")
def test_flatten0():
    array = ak.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])
    array = v1_to_v2(array.layout)

    assert ak.to_list(array.flatten(axis=0)) == [1.1, 2.2, 3.3, 4.4, 5.5]

    content0 = ak.from_iter(
        [1.1, 2.2, None, 3.3, None, None, 4.4, 5.5], highlevel=False
    )
    content1 = ak.from_iter(["one", None, "two", None, "three"], highlevel=False)
    array = ak.Array(
        ak.layout.UnionArray8_64(
            ak.layout.Index8(
                np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=np.int8)
            ),
            ak.layout.Index64(
                np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 4, 7], dtype=np.int64)
            ),
            [content0, content1],
        )
    )
    array = v1_to_v2(array.layout)

    assert ak.to_list(array) == [
        1.1,
        "one",
        2.2,
        None,
        None,
        "two",
        3.3,
        None,
        None,
        None,
        4.4,
        "three",
        5.5,
    ]
    assert ak.to_list(array.flatten(axis=0)) == [
        1.1,
        "one",
        2.2,
        "two",
        3.3,
        4.4,
        "three",
        5.5,
    ]


@pytest.mark.skip(reason="ak.flatten() not implemented yet")
def test_fix_flatten_of_sliced_array():
    array = ak.Array([[1, 2, 3], [], [4, 5], [6, 7, 8, 9]])
    array = v1_to_v2(array.layout)

    assert array[:-1].flatten(axis=1).tolist() == [1, 2, 3, 4, 5]
    assert array[:-2].flatten(axis=1).tolist() == [1, 2, 3]
    assert array[:-1].flatten(axis=None).tolist() == [1, 2, 3, 4, 5]
    assert array[:-2].flatten(axis=None).tolist() == [1, 2, 3]


@pytest.mark.skip(reason="ak.flatten() not implemented yet")
def test_fix_corner_case():
    array = ak.Array([[1, 2, 3], [], [4, 5]])
    array = v1_to_v2(array.layout)

    assert array.flatten(axis=0).tolist() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]
    assert array.flatten(axis=-2).tolist() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]

    array = ak.Array([1, 2, 3, 4, 5])
    array = v1_to_v2(array)

    assert array.flatten(axis=0).tolist() == [1, 2, 3, 4, 5]
    assert array.flatten(axis=-1).tolist() == [1, 2, 3, 4, 5]


@pytest.mark.skip(reason="ak.flatten() not implemented yet")
def test_flatten_allow_regulararray_size_zero():
    empty = ak.Array(
        ak.layout.RegularArray(
            ak.Array([[1, 2, 3], [], [4, 5]]).layout, 0, zeros_length=0
        )
    )
    empty = v1_to_v2(empty.layout)

    assert empty.flatten(axis=0).tolist() == []
    assert empty.flatten(axis=1).tolist() == []
    assert empty.flatten(axis=2).tolist() == []

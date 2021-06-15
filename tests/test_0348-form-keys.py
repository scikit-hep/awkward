# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

try:
    # pybind11 only supports cPickle protocol 2+ (-1 in pickle.dumps)
    # (automatically satisfied in Python 3; this is just to keep testing Python 2.7)
    import cPickle as pickle
except ImportError:
    import pickle

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_numpyarray():
    assert ak.from_buffers(*ak.to_buffers([1, 2, 3, 4, 5])).tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert pickle.loads(pickle.dumps(ak.Array([1, 2, 3, 4, 5]), -1)).tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_listoffsetarray():
    assert ak.from_buffers(*ak.to_buffers([[1, 2, 3], [], [4, 5]])).tolist() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]
    assert ak.from_buffers(
        *ak.to_buffers(["one", "two", "three", "four", "five"])
    ).tolist() == ["one", "two", "three", "four", "five"]
    assert ak.from_buffers(
        *ak.to_buffers([["one", "two", "three"], [], ["four", "five"]])
    ).tolist() == [["one", "two", "three"], [], ["four", "five"]]
    assert pickle.loads(
        pickle.dumps(ak.Array([[1, 2, 3], [], [4, 5]]), -1)
    ).tolist() == [[1, 2, 3], [], [4, 5]]


def test_listarray():
    listoffsetarray = ak.Array([[1, 2, 3], [], [4, 5]]).layout
    listarray = ak.layout.ListArray64(
        listoffsetarray.starts, listoffsetarray.stops, listoffsetarray.content
    )
    assert ak.from_buffers(*ak.to_buffers(listarray)).tolist() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]
    assert pickle.loads(pickle.dumps(ak.Array(listarray), -1)).tolist() == [
        [1, 2, 3],
        [],
        [4, 5],
    ]


def test_indexedoptionarray():
    assert ak.from_buffers(*ak.to_buffers([1, 2, 3, None, None, 5])).tolist() == [
        1,
        2,
        3,
        None,
        None,
        5,
    ]
    assert pickle.loads(
        pickle.dumps(ak.Array([1, 2, 3, None, None, 5]), -1)
    ).tolist() == [1, 2, 3, None, None, 5]


def test_indexedarray():
    content = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    index = ak.layout.Index64(np.array([3, 1, 1, 4, 2], dtype=np.int64))
    indexedarray = ak.layout.IndexedArray64(index, content)
    assert ak.from_buffers(*ak.to_buffers(indexedarray)).tolist() == [
        3.3,
        1.1,
        1.1,
        4.4,
        2.2,
    ]
    assert pickle.loads(pickle.dumps(ak.Array(indexedarray), -1)).tolist() == [
        3.3,
        1.1,
        1.1,
        4.4,
        2.2,
    ]


def test_emptyarray():
    assert ak.from_buffers(*ak.to_buffers([])).tolist() == []
    assert ak.from_buffers(*ak.to_buffers([[], [], []])).tolist() == [[], [], []]

    assert pickle.loads(pickle.dumps(ak.Array([]), -1)).tolist() == []
    assert pickle.loads(pickle.dumps(ak.Array([[], [], []]), -1)).tolist() == [
        [],
        [],
        [],
    ]


def test_bytemaskedarray():
    content = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    mask = ak.layout.Index8(np.array([False, True, True, False, False], dtype=np.int8))
    bytemaskedarray = ak.layout.ByteMaskedArray(mask, content, True)
    assert ak.from_buffers(*ak.to_buffers(bytemaskedarray)).tolist() == [
        None,
        1.1,
        2.2,
        None,
        None,
    ]
    assert pickle.loads(pickle.dumps(ak.Array(bytemaskedarray), -1)).tolist() == [
        None,
        1.1,
        2.2,
        None,
        None,
    ]


def test_bitmaskedarray():
    content = ak.Array([0.0, 1.1, 2.2, 3.3, 4.4]).layout
    mask = ak.layout.IndexU8(
        np.packbits(np.array([False, True, True, False, False], dtype=np.int8))
    )
    bitmaskedarray = ak.layout.BitMaskedArray(mask, content, True, 5, False)
    assert ak.from_buffers(*ak.to_buffers(bitmaskedarray)).tolist() == [
        None,
        1.1,
        2.2,
        None,
        None,
    ]
    assert pickle.loads(pickle.dumps(ak.Array(bitmaskedarray), -1)).tolist() == [
        None,
        1.1,
        2.2,
        None,
        None,
    ]


def test_recordarray():
    assert ak.from_buffers(
        *ak.to_buffers([(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])])
    ).tolist() == [(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])]
    assert ak.from_buffers(
        *ak.to_buffers(
            [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [1, 2]}, {"x": 3.3, "y": [1, 2, 3]}]
        )
    ).tolist() == [
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [1, 2]},
        {"x": 3.3, "y": [1, 2, 3]},
    ]

    assert pickle.loads(
        pickle.dumps(ak.Array([(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])]), -1)
    ).tolist() == [(1.1, [1]), (2.2, [1, 2]), (3.3, [1, 2, 3])]
    assert pickle.loads(
        pickle.dumps(
            ak.Array(
                [
                    {"x": 1.1, "y": [1]},
                    {"x": 2.2, "y": [1, 2]},
                    {"x": 3.3, "y": [1, 2, 3]},
                ]
            ),
            -1,
        )
    ).tolist() == [
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [1, 2]},
        {"x": 3.3, "y": [1, 2, 3]},
    ]


def test_record():
    assert pickle.loads(
        pickle.dumps(ak.Record({"x": 2.2, "y": [1, 2]}), -1)
    ).tolist() == {"x": 2.2, "y": [1, 2]}
    assert (
        pickle.loads(
            pickle.dumps(
                ak.Array(
                    [
                        {"x": 1.1, "y": [1]},
                        {"x": 2.2, "y": [1, 2]},
                        {"x": 3.3, "y": [1, 2, 3]},
                    ]
                )[1],
                -1,
            )
        ).tolist()
        == {"x": 2.2, "y": [1, 2]}
    )


def test_regulararray():
    content = ak.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).layout
    regulararray = ak.layout.RegularArray(content, 3, zeros_length=0)
    assert ak.from_buffers(*ak.to_buffers(regulararray)).tolist() == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    assert pickle.loads(pickle.dumps(ak.Array(regulararray), -1)).tolist() == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]


def test_unionarray():
    assert ak.from_buffers(*ak.to_buffers([[1, 2, 3], [], 4, 5])).tolist() == [
        [1, 2, 3],
        [],
        4,
        5,
    ]
    assert pickle.loads(pickle.dumps(ak.Array([[1, 2, 3], [], 4, 5]), -1)).tolist() == [
        [1, 2, 3],
        [],
        4,
        5,
    ]


def test_unmaskedarray():
    content = ak.Array([1, 2, 3, 4, 5]).layout
    unmaskedarray = ak.layout.UnmaskedArray(content)
    assert ak.from_buffers(*ak.to_buffers(unmaskedarray)).tolist() == [1, 2, 3, 4, 5]
    assert pickle.loads(pickle.dumps(ak.Array(unmaskedarray), -1)).tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_partitioned():
    array = ak.repartition(ak.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 3)

    form, length, container = ak.to_buffers(array)
    assert ak.from_buffers(form, length, container).tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]

    form, length, container = ak.to_buffers(array)
    assert ak.from_buffers(form, length, container).tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]

    one = ak.Array([1, 2, 3, 4, 5])
    two = ak.Array([6, 7, 8, 9, 10])
    container = {}
    lengths = []
    form1, length, _ = ak.to_buffers(one, container, 0)
    lengths.append(length)
    form2, length, _ = ak.to_buffers(two, container, 1)
    lengths.append(length)
    assert form1 == form2

    assert ak.from_buffers(form1, lengths, container).tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]
    assert pickle.loads(pickle.dumps(array, -1)).tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]


def test_lazy():
    array = ak.Array([1, 2, 3, 4, 5])

    form, length, container = ak.to_buffers(array)

    assert ak.from_buffers(form, length, container, lazy=True).tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_lazy_partitioned():
    array = ak.repartition(ak.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 3)
    form, length, container = ak.to_buffers(array)
    assert length == [3, 3, 3, 1]

    assert ak.from_buffers(form, length, container, lazy=True).tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def test_numpyarray_merge():
    emptyarray = ak.layout.EmptyArray()

    np1 = np.arange(2 * 7 * 5).reshape(2, 7, 5)
    np2 = np.arange(3 * 7 * 5).reshape(3, 7, 5)
    ak1 = ak.layout.NumpyArray(np1)
    ak2 = ak.layout.NumpyArray(np2)

    assert ak.to_list(ak1.merge(ak2)) == ak.to_list(np.concatenate([np1, np2]))
    assert ak.to_list(ak1[1:, :-1, ::-1].merge(ak2[1:, :-1, ::-1])) == ak.to_list(
        np.concatenate([np1[1:, :-1, ::-1], np2[1:, :-1, ::-1]])
    )

    for x in [
        np.bool,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ]:
        for y in [
            np.bool,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float32,
            np.float64,
        ]:
            z = np.concatenate(
                [np.array([1, 2, 3], dtype=x), np.array([4, 5], dtype=y)]
            ).dtype.type
            one = ak.layout.NumpyArray(np.array([1, 2, 3], dtype=x))
            two = ak.layout.NumpyArray(np.array([4, 5], dtype=y))
            three = one.merge(two)
            assert np.asarray(three).dtype == np.dtype(z), "{0} {1} {2} {3}".format(
                x, y, z, np.asarray(three).dtype.type
            )
            assert ak.to_list(three) == ak.to_list(
                np.concatenate([np.asarray(one), np.asarray(two)])
            )
            assert ak.to_list(one.merge(emptyarray)) == ak.to_list(one)
            assert ak.to_list(emptyarray.merge(one)) == ak.to_list(one)


def test_regulararray_merge():
    emptyarray = ak.layout.EmptyArray()

    np1 = np.arange(2 * 7 * 5).reshape(2, 7, 5)
    np2 = np.arange(3 * 7 * 5).reshape(3, 7, 5)
    ak1 = ak.from_iter(np1, highlevel=False)
    ak2 = ak.from_iter(np2, highlevel=False)

    assert ak.to_list(ak1.merge(ak2)) == ak.to_list(np.concatenate([np1, np2]))
    assert ak.to_list(ak1.merge(emptyarray)) == ak.to_list(ak1)
    assert ak.to_list(emptyarray.merge(ak1)) == ak.to_list(ak1)


def test_listarray_merge():
    emptyarray = ak.layout.EmptyArray()

    content1 = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    content2 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7]))

    for (dtype1, Index1, ListArray1), (dtype2, Index2, ListArray2) in [
        (
            (np.int32, ak.layout.Index32, ak.layout.ListArray32),
            (np.int32, ak.layout.Index32, ak.layout.ListArray32),
        ),
        (
            (np.int32, ak.layout.Index32, ak.layout.ListArray32),
            (np.uint32, ak.layout.IndexU32, ak.layout.ListArrayU32),
        ),
        (
            (np.int32, ak.layout.Index32, ak.layout.ListArray32),
            (np.int64, ak.layout.Index64, ak.layout.ListArray64),
        ),
        (
            (np.uint32, ak.layout.IndexU32, ak.layout.ListArrayU32),
            (np.int32, ak.layout.Index32, ak.layout.ListArray32),
        ),
        (
            (np.uint32, ak.layout.IndexU32, ak.layout.ListArrayU32),
            (np.uint32, ak.layout.IndexU32, ak.layout.ListArrayU32),
        ),
        (
            (np.uint32, ak.layout.IndexU32, ak.layout.ListArrayU32),
            (np.int64, ak.layout.Index64, ak.layout.ListArray64),
        ),
        (
            (np.int64, ak.layout.Index64, ak.layout.ListArray64),
            (np.int32, ak.layout.Index32, ak.layout.ListArray32),
        ),
        (
            (np.int64, ak.layout.Index64, ak.layout.ListArray64),
            (np.uint32, ak.layout.IndexU32, ak.layout.ListArrayU32),
        ),
        (
            (np.int64, ak.layout.Index64, ak.layout.ListArray64),
            (np.int64, ak.layout.Index64, ak.layout.ListArray64),
        ),
    ]:
        starts1 = Index1(np.array([0, 3, 3], dtype=dtype1))
        stops1 = Index1(np.array([3, 3, 5], dtype=dtype1))
        starts2 = Index2(np.array([2, 99, 0], dtype=dtype2))
        stops2 = Index2(np.array([6, 99, 3], dtype=dtype2))
        array1 = ListArray1(starts1, stops1, content1)
        array2 = ListArray2(starts2, stops2, content2)
        assert ak.to_list(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert ak.to_list(array2) == [[3, 4, 5, 6], [], [1, 2, 3]]

        assert ak.to_list(array1.merge(array2)) == [
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [3, 4, 5, 6],
            [],
            [1, 2, 3],
        ]
        assert ak.to_list(array2.merge(array1)) == [
            [3, 4, 5, 6],
            [],
            [1, 2, 3],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]
        assert ak.to_list(array1.merge(emptyarray)) == ak.to_list(array1)
        assert ak.to_list(emptyarray.merge(array1)) == ak.to_list(array1)

    regulararray = ak.layout.RegularArray(content2, 2)
    assert ak.to_list(regulararray) == [[1, 2], [3, 4], [5, 6]]
    assert ak.to_list(regulararray.merge(emptyarray)) == ak.to_list(regulararray)
    assert ak.to_list(emptyarray.merge(regulararray)) == ak.to_list(regulararray)

    for (dtype1, Index1, ListArray1) in [
        (np.int32, ak.layout.Index32, ak.layout.ListArray32),
        (np.uint32, ak.layout.IndexU32, ak.layout.ListArrayU32),
        (np.int64, ak.layout.Index64, ak.layout.ListArray64),
    ]:
        starts1 = Index1(np.array([0, 3, 3], dtype=dtype1))
        stops1 = Index1(np.array([3, 3, 5], dtype=dtype1))
        array1 = ListArray1(starts1, stops1, content1)

        assert ak.to_list(array1.merge(regulararray)) == [
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        assert ak.to_list(regulararray.merge(array1)) == [
            [1, 2],
            [3, 4],
            [5, 6],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]


def test_listoffsetarray_merge():
    emptyarray = ak.layout.EmptyArray()

    content1 = ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    content2 = ak.layout.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7]))

    for (dtype1, Index1, ListOffsetArray1), (dtype2, Index2, ListOffsetArray2) in [
        (
            (np.int32, ak.layout.Index32, ak.layout.ListOffsetArray32),
            (np.int32, ak.layout.Index32, ak.layout.ListOffsetArray32),
        ),
        (
            (np.int32, ak.layout.Index32, ak.layout.ListOffsetArray32),
            (np.uint32, ak.layout.IndexU32, ak.layout.ListOffsetArrayU32),
        ),
        (
            (np.int32, ak.layout.Index32, ak.layout.ListOffsetArray32),
            (np.int64, ak.layout.Index64, ak.layout.ListOffsetArray64),
        ),
        (
            (np.uint32, ak.layout.IndexU32, ak.layout.ListOffsetArrayU32),
            (np.int32, ak.layout.Index32, ak.layout.ListOffsetArray32),
        ),
        (
            (np.uint32, ak.layout.IndexU32, ak.layout.ListOffsetArrayU32),
            (np.uint32, ak.layout.IndexU32, ak.layout.ListOffsetArrayU32),
        ),
        (
            (np.uint32, ak.layout.IndexU32, ak.layout.ListOffsetArrayU32),
            (np.int64, ak.layout.Index64, ak.layout.ListOffsetArray64),
        ),
        (
            (np.int64, ak.layout.Index64, ak.layout.ListOffsetArray64),
            (np.int32, ak.layout.Index32, ak.layout.ListOffsetArray32),
        ),
        (
            (np.int64, ak.layout.Index64, ak.layout.ListOffsetArray64),
            (np.uint32, ak.layout.IndexU32, ak.layout.ListOffsetArrayU32),
        ),
        (
            (np.int64, ak.layout.Index64, ak.layout.ListOffsetArray64),
            (np.int64, ak.layout.Index64, ak.layout.ListOffsetArray64),
        ),
    ]:
        offsets1 = Index1(np.array([0, 3, 3, 5], dtype=dtype1))
        offsets2 = Index2(np.array([1, 3, 3, 3, 5], dtype=dtype2))
        array1 = ListOffsetArray1(offsets1, content1)
        array2 = ListOffsetArray2(offsets2, content2)
        assert ak.to_list(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert ak.to_list(array2) == [[2, 3], [], [], [4, 5]]

        assert ak.to_list(array1.merge(array2)) == [
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [2, 3],
            [],
            [],
            [4, 5],
        ]
        assert ak.to_list(array2.merge(array1)) == [
            [2, 3],
            [],
            [],
            [4, 5],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]
        assert ak.to_list(array1.merge(emptyarray)) == ak.to_list(array1)
        assert ak.to_list(emptyarray.merge(array1)) == ak.to_list(array1)

    regulararray = ak.layout.RegularArray(content2, 2)
    assert ak.to_list(regulararray) == [[1, 2], [3, 4], [5, 6]]

    for (dtype1, Index1, ListArray1) in [
        (np.int32, ak.layout.Index32, ak.layout.ListArray32),
        (np.uint32, ak.layout.IndexU32, ak.layout.ListArrayU32),
        (np.int64, ak.layout.Index64, ak.layout.ListArray64),
    ]:
        starts1 = Index1(np.array([0, 3, 3], dtype=dtype1))
        stops1 = Index1(np.array([3, 3, 5], dtype=dtype1))
        array1 = ListArray1(starts1, stops1, content1)

        assert ak.to_list(array1.merge(regulararray)) == [
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        assert ak.to_list(regulararray.merge(array1)) == [
            [1, 2],
            [3, 4],
            [5, 6],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]


def test_recordarray_merge():
    emptyarray = ak.layout.EmptyArray()

    arrayr1 = ak.from_iter(
        [{"x": 0, "y": []}, {"x": 1, "y": [1, 1]}, {"x": 2, "y": [2, 2]}],
        highlevel=False,
    )
    arrayr2 = ak.from_iter(
        [
            {"x": 2.2, "y": [2.2, 2.2]},
            {"x": 1.1, "y": [1.1, 1.1]},
            {"x": 0.0, "y": [0.0, 0.0]},
        ],
        highlevel=False,
    )
    arrayr3 = ak.from_iter(
        [{"x": 0, "y": 0.0}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}], highlevel=False
    )
    arrayr4 = ak.from_iter(
        [{"y": [], "x": 0}, {"y": [1, 1], "x": 1}, {"y": [2, 2], "x": 2}],
        highlevel=False,
    )
    arrayr5 = ak.from_iter(
        [
            {"x": 0, "y": [], "z": 0},
            {"x": 1, "y": [1, 1], "z": 1},
            {"x": 2, "y": [2, 2], "z": 2},
        ],
        highlevel=False,
    )
    arrayr6 = ak.from_iter(
        [
            {"z": 0, "x": 0, "y": []},
            {"z": 1, "x": 1, "y": [1, 1]},
            {"z": 2, "x": 2, "y": [2, 2]},
        ],
        highlevel=False,
    )
    arrayr7 = ak.from_iter([{"x": 0}, {"x": 1}, {"x": 2}], highlevel=False)

    arrayt1 = ak.from_iter([(0, []), (1, [1.1]), (2, [2, 2])], highlevel=False)
    arrayt2 = ak.from_iter(
        [(2.2, [2.2, 2.2]), (1.1, [1.1, 1.1]), (0.0, [0.0, 0.0])], highlevel=False
    )
    arrayt3 = ak.from_iter([(0, 0.0), (1, 1.1), (2, 2.2)], highlevel=False)
    arrayt4 = ak.from_iter([([], 0), ([1.1], 1), ([2.2, 2.2], 2)], highlevel=False)
    arrayt5 = ak.from_iter([(0, [], 0), (1, [1], 1), (2, [2, 2], 2)], highlevel=False)
    arrayt6 = ak.from_iter([(0, 0, []), (1, 1, [1]), (2, 2, [2, 2])], highlevel=False)
    arrayt7 = ak.from_iter([(0,), (1,), (2,)], highlevel=False)

    assert arrayr1.mergeable(arrayr2)
    assert arrayr2.mergeable(arrayr1)
    assert not arrayr1.mergeable(arrayr3)
    assert arrayr1.mergeable(arrayr4)
    assert arrayr4.mergeable(arrayr1)
    assert not arrayr1.mergeable(arrayr5)
    assert not arrayr1.mergeable(arrayr6)
    assert arrayr5.mergeable(arrayr6)
    assert arrayr6.mergeable(arrayr5)
    assert not arrayr1.mergeable(arrayr7)

    assert arrayt1.mergeable(arrayt2)
    assert arrayt2.mergeable(arrayt1)
    assert not arrayt1.mergeable(arrayt3)
    assert not arrayt1.mergeable(arrayt4)
    assert not arrayt1.mergeable(arrayt5)
    assert not arrayt1.mergeable(arrayt6)
    assert not arrayt5.mergeable(arrayt6)
    assert not arrayt1.mergeable(arrayt7)

    assert ak.to_list(arrayr1.merge(arrayr2)) == [
        {"x": 0.0, "y": []},
        {"x": 1.0, "y": [1.0, 1.0]},
        {"x": 2.0, "y": [2.0, 2.0]},
        {"x": 2.2, "y": [2.2, 2.2]},
        {"x": 1.1, "y": [1.1, 1.1]},
        {"x": 0.0, "y": [0.0, 0.0]},
    ]
    assert ak.to_list(arrayr2.merge(arrayr1)) == [
        {"x": 2.2, "y": [2.2, 2.2]},
        {"x": 1.1, "y": [1.1, 1.1]},
        {"x": 0.0, "y": [0.0, 0.0]},
        {"x": 0.0, "y": []},
        {"x": 1.0, "y": [1.0, 1.0]},
        {"x": 2.0, "y": [2.0, 2.0]},
    ]

    assert ak.to_list(arrayr1.merge(arrayr4)) == [
        {"x": 0, "y": []},
        {"x": 1, "y": [1, 1]},
        {"x": 2, "y": [2, 2]},
        {"x": 0, "y": []},
        {"x": 1, "y": [1, 1]},
        {"x": 2, "y": [2, 2]},
    ]
    assert ak.to_list(arrayr4.merge(arrayr1)) == [
        {"x": 0, "y": []},
        {"x": 1, "y": [1, 1]},
        {"x": 2, "y": [2, 2]},
        {"x": 0, "y": []},
        {"x": 1, "y": [1, 1]},
        {"x": 2, "y": [2, 2]},
    ]

    assert ak.to_list(arrayr5.merge(arrayr6)) == [
        {"x": 0, "y": [], "z": 0},
        {"x": 1, "y": [1, 1], "z": 1},
        {"x": 2, "y": [2, 2], "z": 2},
        {"x": 0, "y": [], "z": 0},
        {"x": 1, "y": [1, 1], "z": 1},
        {"x": 2, "y": [2, 2], "z": 2},
    ]
    assert ak.to_list(arrayr6.merge(arrayr5)) == [
        {"x": 0, "y": [], "z": 0},
        {"x": 1, "y": [1, 1], "z": 1},
        {"x": 2, "y": [2, 2], "z": 2},
        {"x": 0, "y": [], "z": 0},
        {"x": 1, "y": [1, 1], "z": 1},
        {"x": 2, "y": [2, 2], "z": 2},
    ]

    assert ak.to_list(arrayt1.merge(arrayt2)) == [
        (0.0, []),
        (1.0, [1.1]),
        (2.0, [2.0, 2.0]),
        (2.2, [2.2, 2.2]),
        (1.1, [1.1, 1.1]),
        (0.0, [0.0, 0.0]),
    ]
    assert ak.to_list(arrayt2.merge(arrayt1)) == [
        (2.2, [2.2, 2.2]),
        (1.1, [1.1, 1.1]),
        (0.0, [0.0, 0.0]),
        (0.0, []),
        (1.0, [1.1]),
        (2.0, [2.0, 2.0]),
    ]

    assert ak.to_list(arrayr1.merge(emptyarray)) == ak.to_list(arrayr1)
    assert ak.to_list(arrayr2.merge(emptyarray)) == ak.to_list(arrayr2)
    assert ak.to_list(arrayr3.merge(emptyarray)) == ak.to_list(arrayr3)
    assert ak.to_list(arrayr4.merge(emptyarray)) == ak.to_list(arrayr4)
    assert ak.to_list(arrayr5.merge(emptyarray)) == ak.to_list(arrayr5)
    assert ak.to_list(arrayr6.merge(emptyarray)) == ak.to_list(arrayr6)
    assert ak.to_list(arrayr7.merge(emptyarray)) == ak.to_list(arrayr7)

    assert ak.to_list(emptyarray.merge(arrayr1)) == ak.to_list(arrayr1)
    assert ak.to_list(emptyarray.merge(arrayr2)) == ak.to_list(arrayr2)
    assert ak.to_list(emptyarray.merge(arrayr3)) == ak.to_list(arrayr3)
    assert ak.to_list(emptyarray.merge(arrayr4)) == ak.to_list(arrayr4)
    assert ak.to_list(emptyarray.merge(arrayr5)) == ak.to_list(arrayr5)
    assert ak.to_list(emptyarray.merge(arrayr6)) == ak.to_list(arrayr6)
    assert ak.to_list(emptyarray.merge(arrayr7)) == ak.to_list(arrayr7)

    assert ak.to_list(arrayt1.merge(emptyarray)) == ak.to_list(arrayt1)
    assert ak.to_list(arrayt2.merge(emptyarray)) == ak.to_list(arrayt2)
    assert ak.to_list(arrayt3.merge(emptyarray)) == ak.to_list(arrayt3)
    assert ak.to_list(arrayt4.merge(emptyarray)) == ak.to_list(arrayt4)
    assert ak.to_list(arrayt5.merge(emptyarray)) == ak.to_list(arrayt5)
    assert ak.to_list(arrayt6.merge(emptyarray)) == ak.to_list(arrayt6)
    assert ak.to_list(arrayt7.merge(emptyarray)) == ak.to_list(arrayt7)

    assert ak.to_list(emptyarray.merge(arrayt1)) == ak.to_list(arrayt1)
    assert ak.to_list(emptyarray.merge(arrayt2)) == ak.to_list(arrayt2)
    assert ak.to_list(emptyarray.merge(arrayt3)) == ak.to_list(arrayt3)
    assert ak.to_list(emptyarray.merge(arrayt4)) == ak.to_list(arrayt4)
    assert ak.to_list(emptyarray.merge(arrayt5)) == ak.to_list(arrayt5)
    assert ak.to_list(emptyarray.merge(arrayt6)) == ak.to_list(arrayt6)
    assert ak.to_list(emptyarray.merge(arrayt7)) == ak.to_list(arrayt7)


def test_indexedarray_merge():
    emptyarray = ak.layout.EmptyArray()

    content1 = ak.from_iter([[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False)
    content2 = ak.from_iter([[1, 2], [], [3, 4]], highlevel=False)
    index1 = ak.layout.Index64(np.array([2, 0, -1, 0, 1, 2], dtype=np.int64))
    indexedarray1 = ak.layout.IndexedOptionArray64(index1, content1)
    assert ak.to_list(indexedarray1) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]

    assert ak.to_list(indexedarray1.merge(content2)) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [1.0, 2.0],
        [],
        [3.0, 4.0],
    ]
    assert ak.to_list(content2.merge(indexedarray1)) == [
        [1.0, 2.0],
        [],
        [3.0, 4.0],
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert ak.to_list(indexedarray1.merge(indexedarray1)) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]


def test_unionarray_merge():
    emptyarray = ak.layout.EmptyArray()

    one = ak.from_iter([0.0, 1.1, 2.2, [], [1], [2, 2]], highlevel=False)
    two = ak.from_iter(
        [{"x": 1, "y": 1.1}, 999, 123, {"x": 2, "y": 2.2}], highlevel=False
    )
    three = ak.from_iter(["one", "two", "three"], highlevel=False)

    assert ak.to_list(one.merge(two)) == [
        0.0,
        1.1,
        2.2,
        [],
        [1],
        [2, 2],
        {"x": 1, "y": 1.1},
        999,
        123,
        {"x": 2, "y": 2.2},
    ]
    assert ak.to_list(two.merge(one)) == [
        {"x": 1, "y": 1.1},
        999,
        123,
        {"x": 2, "y": 2.2},
        0.0,
        1.1,
        2.2,
        [],
        [1],
        [2, 2],
    ]

    assert ak.to_list(one.merge(emptyarray)) == [0.0, 1.1, 2.2, [], [1], [2, 2]]
    assert ak.to_list(emptyarray.merge(one)) == [0.0, 1.1, 2.2, [], [1], [2, 2]]

    assert ak.to_list(one.merge(three)) == [
        0.0,
        1.1,
        2.2,
        [],
        [1],
        [2, 2],
        "one",
        "two",
        "three",
    ]
    assert ak.to_list(two.merge(three)) == [
        {"x": 1, "y": 1.1},
        999,
        123,
        {"x": 2, "y": 2.2},
        "one",
        "two",
        "three",
    ]
    assert ak.to_list(three.merge(one)) == [
        "one",
        "two",
        "three",
        0.0,
        1.1,
        2.2,
        [],
        [1],
        [2, 2],
    ]
    assert ak.to_list(three.merge(two)) == [
        "one",
        "two",
        "three",
        {"x": 1, "y": 1.1},
        999,
        123,
        {"x": 2, "y": 2.2},
    ]


def test_merge_parameters():
    one = ak.from_iter(
        [[121, 117, 99, 107, 121], [115, 116, 117, 102, 102]], highlevel=False
    )
    two = ak.from_iter(["good", "stuff"], highlevel=False)

    assert ak.to_list(ak.concatenate([one, two])) == [
        [121, 117, 99, 107, 121],
        [115, 116, 117, 102, 102],
        "good",
        "stuff",
    ]
    assert ak.to_list(ak.concatenate([two, one])) == [
        "good",
        "stuff",
        [121, 117, 99, 107, 121],
        [115, 116, 117, 102, 102],
    ]


def test_bytemask():
    array = ak.from_iter(
        ["one", "two", None, "three", None, None, "four"], highlevel=False
    )
    assert np.asarray(array.bytemask()).tolist() == [0, 0, 1, 0, 1, 1, 0]

    index2 = ak.layout.Index64(np.array([2, 2, 1, 5, 0], dtype=np.int64))
    array2 = ak.layout.IndexedArray64(index2, array)
    assert np.asarray(array2.bytemask()).tolist() == [0, 0, 0, 0, 0]


def test_indexedarray_simplify():
    array = ak.from_iter(
        ["one", "two", None, "three", None, None, "four", "five"], highlevel=False
    )
    assert np.asarray(array.index).tolist() == [0, 1, -1, 2, -1, -1, 3, 4]

    index2 = ak.layout.Index64(np.array([2, 2, 1, 6, 5], dtype=np.int64))
    array2 = ak.layout.IndexedArray64(index2, array)
    assert (
        ak.to_list(array2.simplify())
        == ak.to_list(array2)
        == [None, None, "two", "four", None]
    )


def test_indexedarray_simplify_more():
    content = ak.layout.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )

    index1_32 = ak.layout.Index32(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int32))
    index1_U32 = ak.layout.IndexU32(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.uint32))
    index1_64 = ak.layout.Index64(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int64))
    index2_32 = ak.layout.Index32(np.array([0, 2, 4, 6], dtype=np.int32))
    index2_U32 = ak.layout.IndexU32(np.array([0, 2, 4, 6], dtype=np.uint32))
    index2_64 = ak.layout.Index64(np.array([0, 2, 4, 6], dtype=np.int64))

    array = ak.layout.IndexedArray32(
        index2_32, ak.layout.IndexedArray32(index1_32, content)
    )
    assert ak.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArray32(
        index2_32, ak.layout.IndexedArrayU32(index1_U32, content)
    )
    assert ak.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArray32(
        index2_32, ak.layout.IndexedArray64(index1_64, content)
    )
    assert ak.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArrayU32(
        index2_U32, ak.layout.IndexedArray32(index1_32, content)
    )
    assert ak.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArrayU32(
        index2_U32, ak.layout.IndexedArrayU32(index1_U32, content)
    )
    assert ak.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArrayU32(
        index2_U32, ak.layout.IndexedArray64(index1_64, content)
    )
    assert ak.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArray64(
        index2_64, ak.layout.IndexedArray32(index1_32, content)
    )
    assert ak.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArray64(
        index2_64, ak.layout.IndexedArrayU32(index1_U32, content)
    )
    assert ak.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArray64(
        index2_64, ak.layout.IndexedArray64(index1_64, content)
    )
    assert ak.to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    index1_32 = ak.layout.Index32(np.array([6, 5, -1, 3, -1, 1, 0], dtype=np.int32))
    index1_64 = ak.layout.Index64(np.array([6, 5, -1, 3, -1, 1, 0], dtype=np.int64))
    index2_32 = ak.layout.Index32(np.array([0, 2, 4, 6], dtype=np.int32))
    index2_U32 = ak.layout.IndexU32(np.array([0, 2, 4, 6], dtype=np.uint32))
    index2_64 = ak.layout.Index64(np.array([0, 2, 4, 6], dtype=np.int64))

    array = ak.layout.IndexedArray32(
        index2_32, ak.layout.IndexedOptionArray32(index1_32, content)
    )
    assert ak.to_list(array) == [6.6, None, None, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArray32(
        index2_32, ak.layout.IndexedOptionArray64(index1_64, content)
    )
    assert ak.to_list(array) == [6.6, None, None, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArrayU32(
        index2_U32, ak.layout.IndexedOptionArray32(index1_32, content)
    )
    assert ak.to_list(array) == [6.6, None, None, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArrayU32(
        index2_U32, ak.layout.IndexedOptionArray64(index1_64, content)
    )
    assert ak.to_list(array) == [6.6, None, None, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArray64(
        index2_64, ak.layout.IndexedOptionArray32(index1_32, content)
    )
    assert ak.to_list(array) == [6.6, None, None, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedArray64(
        index2_64, ak.layout.IndexedOptionArray64(index1_64, content)
    )
    assert ak.to_list(array) == [6.6, None, None, 0.0]
    assert ak.to_list(array.simplify()) == [6.6, None, None, 0.0]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    index1_32 = ak.layout.Index32(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int32))
    index1_U32 = ak.layout.IndexU32(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.uint32))
    index1_64 = ak.layout.Index64(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int64))
    index2_32 = ak.layout.Index32(np.array([0, -1, 4, -1], dtype=np.int32))
    index2_64 = ak.layout.Index64(np.array([0, -1, 4, -1], dtype=np.int64))

    array = ak.layout.IndexedOptionArray32(
        index2_32, ak.layout.IndexedArray32(index1_32, content)
    )
    assert ak.to_list(array) == [6.6, None, 2.2, None]
    assert ak.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedOptionArray32(
        index2_32, ak.layout.IndexedArrayU32(index1_U32, content)
    )
    assert ak.to_list(array) == [6.6, None, 2.2, None]
    assert ak.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedOptionArray32(
        index2_32, ak.layout.IndexedArray64(index1_64, content)
    )
    assert ak.to_list(array) == [6.6, None, 2.2, None]
    assert ak.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedOptionArray64(
        index2_64, ak.layout.IndexedArray32(index1_32, content)
    )
    assert ak.to_list(array) == [6.6, None, 2.2, None]
    assert ak.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedOptionArray64(
        index2_64, ak.layout.IndexedArrayU32(index1_U32, content)
    )
    assert ak.to_list(array) == [6.6, None, 2.2, None]
    assert ak.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedOptionArray64(
        index2_64, ak.layout.IndexedArray64(index1_64, content)
    )
    assert ak.to_list(array) == [6.6, None, 2.2, None]
    assert ak.to_list(array.simplify()) == [6.6, None, 2.2, None]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    index1_32 = ak.layout.Index32(np.array([6, 5, -1, 3, -1, 1, 0], dtype=np.int32))
    index1_64 = ak.layout.Index64(np.array([6, 5, -1, 3, -1, 1, 0], dtype=np.int64))
    index2_32 = ak.layout.Index32(np.array([0, -1, 4, -1], dtype=np.int32))
    index2_64 = ak.layout.Index64(np.array([0, -1, 4, -1], dtype=np.int64))

    array = ak.layout.IndexedOptionArray32(
        index2_32, ak.layout.IndexedOptionArray32(index1_32, content)
    )
    assert ak.to_list(array) == [6.6, None, None, None]
    assert ak.to_list(array.simplify()) == [6.6, None, None, None]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedOptionArray32(
        index2_32, ak.layout.IndexedOptionArray64(index1_64, content)
    )
    assert ak.to_list(array) == [6.6, None, None, None]
    assert ak.to_list(array.simplify()) == [6.6, None, None, None]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedOptionArray64(
        index2_64, ak.layout.IndexedOptionArray32(index1_32, content)
    )
    assert ak.to_list(array) == [6.6, None, None, None]
    assert ak.to_list(array.simplify()) == [6.6, None, None, None]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)

    array = ak.layout.IndexedOptionArray64(
        index2_64, ak.layout.IndexedOptionArray64(index1_64, content)
    )
    assert ak.to_list(array) == [6.6, None, None, None]
    assert ak.to_list(array.simplify()) == [6.6, None, None, None]
    assert isinstance(array.simplify(), ak.layout.IndexedOptionArray64)
    assert isinstance(array.simplify().content, ak.layout.NumpyArray)


def test_unionarray_simplify_one():
    one = ak.from_iter([5, 4, 3, 2, 1], highlevel=False)
    two = ak.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    three = ak.from_iter([1.1, 2.2, 3.3], highlevel=False)
    tags = ak.layout.Index8(
        np.array([0, 0, 1, 2, 1, 0, 2, 1, 1, 0, 2, 0], dtype=np.int8)
    )
    index = ak.layout.Index64(
        np.array([0, 1, 0, 0, 1, 2, 1, 2, 3, 3, 2, 4], dtype=np.int64)
    )
    array = ak.layout.UnionArray8_64(tags, index, [one, two, three])

    assert ak.to_list(array) == [
        5,
        4,
        [],
        1.1,
        [1],
        3,
        2.2,
        [2, 2],
        [3, 3, 3],
        2,
        3.3,
        1,
    ]
    assert ak.to_list(array.simplify()) == [
        5.0,
        4.0,
        [],
        1.1,
        [1],
        3.0,
        2.2,
        [2, 2],
        [3, 3, 3],
        2.0,
        3.3,
        1.0,
    ]
    assert len(array.contents) == 3
    assert len(array.simplify().contents) == 2


def test_unionarray_simplify():
    one = ak.from_iter([5, 4, 3, 2, 1], highlevel=False)
    two = ak.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    three = ak.from_iter([1.1, 2.2, 3.3], highlevel=False)

    tags2 = ak.layout.Index8(np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int8))
    index2 = ak.layout.Index32(np.array([0, 0, 1, 1, 2, 3, 2], dtype=np.int32))
    inner = ak.layout.UnionArray8_32(tags2, index2, [two, three])
    tags1 = ak.layout.Index8(
        np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.int8)
    )
    index1 = ak.layout.Index64(
        np.array([0, 1, 0, 1, 2, 2, 3, 4, 5, 3, 6, 4], dtype=np.int64)
    )
    outer = ak.layout.UnionArray8_64(tags1, index1, [one, inner])
    assert ak.to_list(outer) == [
        5,
        4,
        [],
        1.1,
        [1],
        3,
        2.2,
        [2, 2],
        [3, 3, 3],
        2,
        3.3,
        1,
    ]

    assert ak.to_list(outer.simplify()) == [
        5.0,
        4.0,
        [],
        1.1,
        [1],
        3.0,
        2.2,
        [2, 2],
        [3, 3, 3],
        2.0,
        3.3,
        1.0,
    ]
    assert isinstance(outer.content(1), ak.layout.UnionArray8_32)
    assert isinstance(outer.simplify().content(0), ak.layout.NumpyArray)
    assert isinstance(outer.simplify().content(1), ak.layout.ListOffsetArray64)
    assert len(outer.simplify().contents) == 2

    tags2 = ak.layout.Index8(np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int8))
    index2 = ak.layout.Index64(np.array([0, 0, 1, 1, 2, 3, 2], dtype=np.int64))
    inner = ak.layout.UnionArray8_64(tags2, index2, [two, three])
    tags1 = ak.layout.Index8(
        np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=np.int8)
    )
    index1 = ak.layout.Index32(
        np.array([0, 1, 0, 1, 2, 2, 3, 4, 5, 3, 6, 4], dtype=np.int32)
    )
    outer = ak.layout.UnionArray8_32(tags1, index1, [inner, one])
    assert ak.to_list(outer) == [
        5,
        4,
        [],
        1.1,
        [1],
        3,
        2.2,
        [2, 2],
        [3, 3, 3],
        2,
        3.3,
        1,
    ]


def test_concatenate():
    one = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True)
    two = ak.Array([[], [1], [2, 2], [3, 3, 3]], check_valid=True)
    three = ak.Array([True, False, False, True, True], check_valid=True)

    assert ak.to_list(ak.concatenate([one, two, three])) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        [],
        [1],
        [2, 2],
        [3, 3, 3],
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
    ]
    assert isinstance(
        ak.concatenate([one, two, three], highlevel=False), ak.layout.UnionArray8_64
    )
    assert len(ak.concatenate([one, two, three], highlevel=False).contents) == 2


def test_where():
    condition = ak.Array([True, False, True, False, True], check_valid=True)
    one = ak.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True)
    two = ak.Array([False, False, False, True, True], check_valid=True)
    three = ak.Array([[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4]], check_valid=True)

    assert ak.to_list(ak.where(condition, one, two)) == [1.1, 0.0, 3.3, 1.0, 5.5]
    assert ak.to_list(ak.where(condition, one, three)) == [
        1.1,
        [1],
        3.3,
        [3, 3, 3],
        5.5,
    ]

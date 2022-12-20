# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import numpy as np
import pytest  # noqa: F401

import awkward as ak

to_list = ak.operations.to_list


def test_numpyarray_merge():
    emptyarray = ak.contents.EmptyArray()

    np1 = np.arange(2 * 7 * 5).reshape(2, 7, 5)
    np2 = np.arange(3 * 7 * 5).reshape(3, 7, 5)
    ak1 = ak.contents.NumpyArray(np1)
    ak2 = ak.contents.NumpyArray(np2)
    assert to_list(ak1._mergemany([ak2])) == to_list(np.concatenate([np1, np2]))
    assert to_list(ak1[1:, :-1, ::-1]._mergemany([ak2[1:, :-1, ::-1]])) == to_list(
        np.concatenate([np1[1:, :-1, ::-1], np2[1:, :-1, ::-1]])
    )
    assert ak1.to_typetracer()._mergemany([ak2]).form == ak1._mergemany([ak2]).form
    assert (
        ak1[1:, :-1, ::-1].to_typetracer()._mergemany([ak2[1:, :-1, ::-1]]).form
        == ak1[1:, :-1, ::-1]._mergemany([ak2[1:, :-1, ::-1]]).form
    )

    for x in [
        np.bool_,
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
            np.bool_,
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
            one = ak.contents.NumpyArray(np.array([1, 2, 3], dtype=x))
            two = ak.contents.NumpyArray(np.array([4, 5], dtype=y))
            three = one._mergemany([two])
            assert np.asarray(three).dtype == np.dtype(z), "{} {} {} {}".format(
                x, y, z, np.asarray(three).dtype.type
            )
            assert to_list(three) == to_list(
                np.concatenate([np.asarray(one), np.asarray(two)])
            )
            assert to_list(one._mergemany([emptyarray])) == to_list(one)
            assert to_list(emptyarray._mergemany([one])) == to_list(one)

            assert (
                one.to_typetracer()._mergemany([two]).form == one._mergemany([two]).form
            )
            assert (
                one.to_typetracer()._mergemany([emptyarray]).form
                == one._mergemany([emptyarray]).form
            )
            assert (
                emptyarray.to_typetracer()._mergemany([one]).form
                == emptyarray._mergemany([one]).form
            )


def test_regulararray_merge():
    emptyarray = ak.contents.EmptyArray()

    np1 = np.arange(2 * 7 * 5).reshape(2, 7, 5)
    np2 = np.arange(3 * 7 * 5).reshape(3, 7, 5)
    ak1 = ak.operations.from_iter(np1, highlevel=False)
    ak2 = ak.operations.from_iter(np2, highlevel=False)

    assert to_list(ak1._mergemany([ak2])) == to_list(np.concatenate([np1, np2]))
    assert to_list(ak1._mergemany([emptyarray])) == to_list(ak1)
    assert to_list(emptyarray._mergemany([ak1])) == to_list(ak1)

    assert ak1.to_typetracer()._mergemany([ak2]).form == ak1._mergemany([ak2]).form
    assert (
        ak1.to_typetracer()._mergemany([emptyarray]).form
        == ak1._mergemany([emptyarray]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([ak1]).form
        == emptyarray._mergemany([ak1]).form
    )


def test_listarray_merge():
    emptyarray = ak.contents.EmptyArray()

    content1 = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    content2 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7]))

    for (dtype1, Index1, ListArray1), (dtype2, Index2, ListArray2) in [
        (
            (np.int32, ak.index.Index32, ak.contents.ListArray),
            (np.int32, ak.index.Index32, ak.contents.ListArray),
        ),
        (
            (np.int32, ak.index.Index32, ak.contents.ListArray),
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
        ),
        (
            (np.int32, ak.index.Index32, ak.contents.ListArray),
            (np.int64, ak.index.Index64, ak.contents.ListArray),
        ),
        (
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
            (np.int32, ak.index.Index32, ak.contents.ListArray),
        ),
        (
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
        ),
        (
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
            (np.int64, ak.index.Index64, ak.contents.ListArray),
        ),
        (
            (np.int64, ak.index.Index64, ak.contents.ListArray),
            (np.int32, ak.index.Index32, ak.contents.ListArray),
        ),
        (
            (np.int64, ak.index.Index64, ak.contents.ListArray),
            (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
        ),
        (
            (np.int64, ak.index.Index64, ak.contents.ListArray),
            (np.int64, ak.index.Index64, ak.contents.ListArray),
        ),
    ]:
        starts1 = Index1(np.array([0, 3, 3], dtype=dtype1))
        stops1 = Index1(np.array([3, 3, 5], dtype=dtype1))
        starts2 = Index2(np.array([2, 99, 0], dtype=dtype2))
        stops2 = Index2(np.array([6, 99, 3], dtype=dtype2))
        array1 = ListArray1(starts1, stops1, content1)
        array2 = ListArray2(starts2, stops2, content2)
        assert to_list(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert to_list(array2) == [[3, 4, 5, 6], [], [1, 2, 3]]

        assert to_list(array1._mergemany([array2])) == [
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [3, 4, 5, 6],
            [],
            [1, 2, 3],
        ]
        assert to_list(array2._mergemany([array1])) == [
            [3, 4, 5, 6],
            [],
            [1, 2, 3],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]
        assert to_list(array1._mergemany([emptyarray])) == to_list(array1)
        assert to_list(emptyarray._mergemany([array1])) == to_list(array1)

        assert (
            array1.to_typetracer()._mergemany([array2]).form
            == array1._mergemany([array2]).form
        )
        assert (
            array2.to_typetracer()._mergemany([array1]).form
            == array2._mergemany([array1]).form
        )
        assert (
            array1.to_typetracer()._mergemany([emptyarray]).form
            == array1._mergemany([emptyarray]).form
        )
        assert (
            emptyarray.to_typetracer()._mergemany([array1]).form
            == emptyarray._mergemany([array1]).form
        )

    regulararray = ak.contents.RegularArray(content2, 2, zeros_length=0)
    assert to_list(regulararray) == [[1, 2], [3, 4], [5, 6]]
    assert to_list(regulararray._mergemany([emptyarray])) == to_list(regulararray)
    assert to_list(emptyarray._mergemany([regulararray])) == to_list(regulararray)

    for (dtype1, Index1, ListArray1) in [
        (np.int32, ak.index.Index32, ak.contents.ListArray),
        (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
        (np.int64, ak.index.Index64, ak.contents.ListArray),
    ]:
        starts1 = Index1(np.array([0, 3, 3], dtype=dtype1))
        stops1 = Index1(np.array([3, 3, 5], dtype=dtype1))
        array1 = ListArray1(starts1, stops1, content1)

        assert to_list(array1._mergemany([regulararray])) == [
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        assert to_list(regulararray._mergemany([array1])) == [
            [1, 2],
            [3, 4],
            [5, 6],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]


def test_listoffsetarray_merge():
    emptyarray = ak.contents.EmptyArray()

    content1 = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    content2 = ak.contents.NumpyArray(np.array([1, 2, 3, 4, 5, 6, 7]))

    for (dtype1, Index1, ListOffsetArray1), (dtype2, Index2, ListOffsetArray2) in [
        (
            (np.int32, ak.index.Index32, ak.contents.ListOffsetArray),
            (np.int32, ak.index.Index32, ak.contents.ListOffsetArray),
        ),
        (
            (np.int32, ak.index.Index32, ak.contents.ListOffsetArray),
            (np.uint32, ak.index.IndexU32, ak.contents.ListOffsetArray),
        ),
        (
            (np.int32, ak.index.Index32, ak.contents.ListOffsetArray),
            (np.int64, ak.index.Index64, ak.contents.ListOffsetArray),
        ),
        (
            (np.uint32, ak.index.IndexU32, ak.contents.ListOffsetArray),
            (np.int32, ak.index.Index32, ak.contents.ListOffsetArray),
        ),
        (
            (np.uint32, ak.index.IndexU32, ak.contents.ListOffsetArray),
            (np.uint32, ak.index.IndexU32, ak.contents.ListOffsetArray),
        ),
        (
            (np.uint32, ak.index.IndexU32, ak.contents.ListOffsetArray),
            (np.int64, ak.index.Index64, ak.contents.ListOffsetArray),
        ),
        (
            (np.int64, ak.index.Index64, ak.contents.ListOffsetArray),
            (np.int32, ak.index.Index32, ak.contents.ListOffsetArray),
        ),
        (
            (np.int64, ak.index.Index64, ak.contents.ListOffsetArray),
            (np.uint32, ak.index.IndexU32, ak.contents.ListOffsetArray),
        ),
        (
            (np.int64, ak.index.Index64, ak.contents.ListOffsetArray),
            (np.int64, ak.index.Index64, ak.contents.ListOffsetArray),
        ),
    ]:
        offsets1 = Index1(np.array([0, 3, 3, 5], dtype=dtype1))
        offsets2 = Index2(np.array([1, 3, 3, 3, 5], dtype=dtype2))
        array1 = ListOffsetArray1(offsets1, content1)
        array2 = ListOffsetArray2(offsets2, content2)
        assert to_list(array1) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]
        assert to_list(array2) == [[2, 3], [], [], [4, 5]]

        assert to_list(array1._mergemany([array2])) == [
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [2, 3],
            [],
            [],
            [4, 5],
        ]
        assert to_list(array2._mergemany([array1])) == [
            [2, 3],
            [],
            [],
            [4, 5],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]
        assert to_list(array1._mergemany([emptyarray])) == to_list(array1)
        assert to_list(emptyarray._mergemany([array1])) == to_list(array1)

        assert (
            array1.to_typetracer()._mergemany([array2]).form
            == array1._mergemany([array2]).form
        )
        assert (
            array2.to_typetracer()._mergemany([array1]).form
            == array2._mergemany([array1]).form
        )
        assert (
            array1.to_typetracer()._mergemany([emptyarray]).form
            == array1._mergemany([emptyarray]).form
        )
        assert (
            emptyarray.to_typetracer()._mergemany([array1]).form
            == emptyarray._mergemany([array1]).form
        )

    regulararray = ak.contents.RegularArray(content2, 2, zeros_length=0)
    assert to_list(regulararray) == [[1, 2], [3, 4], [5, 6]]

    for (dtype1, Index1, ListArray1) in [
        (np.int32, ak.index.Index32, ak.contents.ListArray),
        (np.uint32, ak.index.IndexU32, ak.contents.ListArray),
        (np.int64, ak.index.Index64, ak.contents.ListArray),
    ]:
        starts1 = Index1(np.array([0, 3, 3], dtype=dtype1))
        stops1 = Index1(np.array([3, 3, 5], dtype=dtype1))
        array1 = ListArray1(starts1, stops1, content1)

        assert to_list(array1._mergemany([regulararray])) == [
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        assert to_list(regulararray._mergemany([array1])) == [
            [1, 2],
            [3, 4],
            [5, 6],
            [1.1, 2.2, 3.3],
            [],
            [4.4, 5.5],
        ]

        assert (
            array1.to_typetracer()._mergemany([regulararray]).form
            == array1._mergemany([regulararray]).form
        )
        assert (
            regulararray.to_typetracer()._mergemany([array1]).form
            == regulararray._mergemany([array1]).form
        )


def test_recordarray_merge():
    emptyarray = ak.contents.EmptyArray()

    arrayr1 = ak.operations.from_iter(
        [{"x": 0, "y": []}, {"x": 1, "y": [1, 1]}, {"x": 2, "y": [2, 2]}],
        highlevel=False,
    )
    arrayr2 = ak.operations.from_iter(
        [
            {"x": 2.2, "y": [2.2, 2.2]},
            {"x": 1.1, "y": [1.1, 1.1]},
            {"x": 0.0, "y": [0.0, 0.0]},
        ],
        highlevel=False,
    )
    arrayr3 = ak.operations.from_iter(
        [{"x": 0, "y": 0.0}, {"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}],
        highlevel=False,
    )
    arrayr4 = ak.operations.from_iter(
        [{"y": [], "x": 0}, {"y": [1, 1], "x": 1}, {"y": [2, 2], "x": 2}],
        highlevel=False,
    )
    arrayr5 = ak.operations.from_iter(
        [
            {"x": 0, "y": [], "z": 0},
            {"x": 1, "y": [1, 1], "z": 1},
            {"x": 2, "y": [2, 2], "z": 2},
        ],
        highlevel=False,
    )
    arrayr6 = ak.operations.from_iter(
        [
            {"z": 0, "x": 0, "y": []},
            {"z": 1, "x": 1, "y": [1, 1]},
            {"z": 2, "x": 2, "y": [2, 2]},
        ],
        highlevel=False,
    )
    arrayr7 = ak.operations.from_iter([{"x": 0}, {"x": 1}, {"x": 2}], highlevel=False)

    arrayt1 = ak.operations.from_iter(
        [(0, []), (1, [1.1]), (2, [2, 2])], highlevel=False
    )
    arrayt2 = ak.operations.from_iter(
        [(2.2, [2.2, 2.2]), (1.1, [1.1, 1.1]), (0.0, [0.0, 0.0])], highlevel=False
    )
    arrayt3 = ak.operations.from_iter([(0, 0.0), (1, 1.1), (2, 2.2)], highlevel=False)
    arrayt4 = ak.operations.from_iter(
        [([], 0), ([1.1], 1), ([2.2, 2.2], 2)], highlevel=False
    )
    arrayt5 = ak.operations.from_iter(
        [(0, [], 0), (1, [1], 1), (2, [2, 2], 2)], highlevel=False
    )
    arrayt6 = ak.operations.from_iter(
        [(0, 0, []), (1, 1, [1]), (2, 2, [2, 2])], highlevel=False
    )
    arrayt7 = ak.operations.from_iter([(0,), (1,), (2,)], highlevel=False)

    assert ak._do.mergeable(arrayr1, arrayr2)
    assert ak._do.mergeable(arrayr2, arrayr1)
    assert not ak._do.mergeable(arrayr1, arrayr3)
    assert ak._do.mergeable(arrayr1, arrayr4)
    assert ak._do.mergeable(arrayr4, arrayr1)
    assert not ak._do.mergeable(arrayr1, arrayr5)
    assert not ak._do.mergeable(arrayr1, arrayr6)
    assert ak._do.mergeable(arrayr5, arrayr6)
    assert ak._do.mergeable(arrayr6, arrayr5)
    assert not ak._do.mergeable(arrayr1, arrayr7)

    assert ak._do.mergeable(arrayt1, arrayt2)
    assert ak._do.mergeable(arrayt2, arrayt1)
    assert not ak._do.mergeable(arrayt1, arrayt3)
    assert not ak._do.mergeable(arrayt1, arrayt4)
    assert not ak._do.mergeable(arrayt1, arrayt5)
    assert not ak._do.mergeable(arrayt1, arrayt6)
    assert not ak._do.mergeable(arrayt5, arrayt6)
    assert not ak._do.mergeable(arrayt1, arrayt7)

    assert to_list(arrayr1._mergemany([arrayr2])) == [
        {"x": 0.0, "y": []},
        {"x": 1.0, "y": [1.0, 1.0]},
        {"x": 2.0, "y": [2.0, 2.0]},
        {"x": 2.2, "y": [2.2, 2.2]},
        {"x": 1.1, "y": [1.1, 1.1]},
        {"x": 0.0, "y": [0.0, 0.0]},
    ]
    assert to_list(arrayr2._mergemany([arrayr1])) == [
        {"x": 2.2, "y": [2.2, 2.2]},
        {"x": 1.1, "y": [1.1, 1.1]},
        {"x": 0.0, "y": [0.0, 0.0]},
        {"x": 0.0, "y": []},
        {"x": 1.0, "y": [1.0, 1.0]},
        {"x": 2.0, "y": [2.0, 2.0]},
    ]

    assert to_list(arrayr1._mergemany([arrayr4])) == [
        {"x": 0, "y": []},
        {"x": 1, "y": [1, 1]},
        {"x": 2, "y": [2, 2]},
        {"x": 0, "y": []},
        {"x": 1, "y": [1, 1]},
        {"x": 2, "y": [2, 2]},
    ]
    assert to_list(arrayr4._mergemany([arrayr1])) == [
        {"x": 0, "y": []},
        {"x": 1, "y": [1, 1]},
        {"x": 2, "y": [2, 2]},
        {"x": 0, "y": []},
        {"x": 1, "y": [1, 1]},
        {"x": 2, "y": [2, 2]},
    ]

    assert to_list(arrayr5._mergemany([arrayr6])) == [
        {"x": 0, "y": [], "z": 0},
        {"x": 1, "y": [1, 1], "z": 1},
        {"x": 2, "y": [2, 2], "z": 2},
        {"x": 0, "y": [], "z": 0},
        {"x": 1, "y": [1, 1], "z": 1},
        {"x": 2, "y": [2, 2], "z": 2},
    ]
    assert to_list(arrayr6._mergemany([arrayr5])) == [
        {"x": 0, "y": [], "z": 0},
        {"x": 1, "y": [1, 1], "z": 1},
        {"x": 2, "y": [2, 2], "z": 2},
        {"x": 0, "y": [], "z": 0},
        {"x": 1, "y": [1, 1], "z": 1},
        {"x": 2, "y": [2, 2], "z": 2},
    ]

    assert to_list(arrayt1._mergemany([arrayt2])) == [
        (0.0, []),
        (1.0, [1.1]),
        (2.0, [2.0, 2.0]),
        (2.2, [2.2, 2.2]),
        (1.1, [1.1, 1.1]),
        (0.0, [0.0, 0.0]),
    ]
    assert to_list(arrayt2._mergemany([arrayt1])) == [
        (2.2, [2.2, 2.2]),
        (1.1, [1.1, 1.1]),
        (0.0, [0.0, 0.0]),
        (0.0, []),
        (1.0, [1.1]),
        (2.0, [2.0, 2.0]),
    ]

    assert (
        arrayr1.to_typetracer()._mergemany([arrayr2]).form
        == arrayr1._mergemany([arrayr2]).form
    )
    assert (
        arrayr2.to_typetracer()._mergemany([arrayr1]).form
        == arrayr2._mergemany([arrayr1]).form
    )
    assert (
        arrayr1.to_typetracer()._mergemany([arrayr4]).form
        == arrayr1._mergemany([arrayr4]).form
    )
    assert (
        arrayr4.to_typetracer()._mergemany([arrayr1]).form
        == arrayr4._mergemany([arrayr1]).form
    )
    assert (
        arrayr5.to_typetracer()._mergemany([arrayr6]).form
        == arrayr5._mergemany([arrayr6]).form
    )
    assert (
        arrayr6.to_typetracer()._mergemany([arrayr5]).form
        == arrayr6._mergemany([arrayr5]).form
    )
    assert (
        arrayt1.to_typetracer()._mergemany([arrayt2]).form
        == arrayt1._mergemany([arrayt2]).form
    )
    assert (
        arrayt2.to_typetracer()._mergemany([arrayt1]).form
        == arrayt2._mergemany([arrayt1]).form
    )

    assert to_list(arrayr1._mergemany([emptyarray])) == to_list(arrayr1)
    assert to_list(arrayr2._mergemany([emptyarray])) == to_list(arrayr2)
    assert to_list(arrayr3._mergemany([emptyarray])) == to_list(arrayr3)
    assert to_list(arrayr4._mergemany([emptyarray])) == to_list(arrayr4)
    assert to_list(arrayr5._mergemany([emptyarray])) == to_list(arrayr5)
    assert to_list(arrayr6._mergemany([emptyarray])) == to_list(arrayr6)
    assert to_list(arrayr7._mergemany([emptyarray])) == to_list(arrayr7)

    assert to_list(emptyarray._mergemany([arrayr1])) == to_list(arrayr1)
    assert to_list(emptyarray._mergemany([arrayr2])) == to_list(arrayr2)
    assert to_list(emptyarray._mergemany([arrayr3])) == to_list(arrayr3)
    assert to_list(emptyarray._mergemany([arrayr4])) == to_list(arrayr4)
    assert to_list(emptyarray._mergemany([arrayr5])) == to_list(arrayr5)
    assert to_list(emptyarray._mergemany([arrayr6])) == to_list(arrayr6)
    assert to_list(emptyarray._mergemany([arrayr7])) == to_list(arrayr7)

    assert to_list(arrayt1._mergemany([emptyarray])) == to_list(arrayt1)
    assert to_list(arrayt2._mergemany([emptyarray])) == to_list(arrayt2)
    assert to_list(arrayt3._mergemany([emptyarray])) == to_list(arrayt3)
    assert to_list(arrayt4._mergemany([emptyarray])) == to_list(arrayt4)
    assert to_list(arrayt5._mergemany([emptyarray])) == to_list(arrayt5)
    assert to_list(arrayt6._mergemany([emptyarray])) == to_list(arrayt6)
    assert to_list(arrayt7._mergemany([emptyarray])) == to_list(arrayt7)

    assert to_list(emptyarray._mergemany([arrayt1])) == to_list(arrayt1)
    assert to_list(emptyarray._mergemany([arrayt2])) == to_list(arrayt2)
    assert to_list(emptyarray._mergemany([arrayt3])) == to_list(arrayt3)
    assert to_list(emptyarray._mergemany([arrayt4])) == to_list(arrayt4)
    assert to_list(emptyarray._mergemany([arrayt5])) == to_list(arrayt5)
    assert to_list(emptyarray._mergemany([arrayt6])) == to_list(arrayt6)
    assert to_list(emptyarray._mergemany([arrayt7])) == to_list(arrayt7)

    assert (
        arrayr1.to_typetracer()._mergemany([emptyarray]).form
        == arrayr1._mergemany([emptyarray]).form
    )
    assert (
        arrayr2.to_typetracer()._mergemany([emptyarray]).form
        == arrayr2._mergemany([emptyarray]).form
    )
    assert (
        arrayr3.to_typetracer()._mergemany([emptyarray]).form
        == arrayr3._mergemany([emptyarray]).form
    )
    assert (
        arrayr4.to_typetracer()._mergemany([emptyarray]).form
        == arrayr4._mergemany([emptyarray]).form
    )
    assert (
        arrayr5.to_typetracer()._mergemany([emptyarray]).form
        == arrayr5._mergemany([emptyarray]).form
    )
    assert (
        arrayr6.to_typetracer()._mergemany([emptyarray]).form
        == arrayr6._mergemany([emptyarray]).form
    )
    assert (
        arrayr7.to_typetracer()._mergemany([emptyarray]).form
        == arrayr7._mergemany([emptyarray]).form
    )

    assert (
        emptyarray.to_typetracer()._mergemany([arrayr1]).form
        == emptyarray._mergemany([arrayr1]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayr2]).form
        == emptyarray._mergemany([arrayr2]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayr3]).form
        == emptyarray._mergemany([arrayr3]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayr4]).form
        == emptyarray._mergemany([arrayr4]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayr5]).form
        == emptyarray._mergemany([arrayr5]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayr6]).form
        == emptyarray._mergemany([arrayr6]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayr7]).form
        == emptyarray._mergemany([arrayr7]).form
    )

    assert (
        arrayt1.to_typetracer()._mergemany([emptyarray]).form
        == arrayt1._mergemany([emptyarray]).form
    )
    assert (
        arrayt2.to_typetracer()._mergemany([emptyarray]).form
        == arrayt2._mergemany([emptyarray]).form
    )
    assert (
        arrayt3.to_typetracer()._mergemany([emptyarray]).form
        == arrayt3._mergemany([emptyarray]).form
    )
    assert (
        arrayt4.to_typetracer()._mergemany([emptyarray]).form
        == arrayt4._mergemany([emptyarray]).form
    )
    assert (
        arrayt5.to_typetracer()._mergemany([emptyarray]).form
        == arrayt5._mergemany([emptyarray]).form
    )
    assert (
        arrayt6.to_typetracer()._mergemany([emptyarray]).form
        == arrayt6._mergemany([emptyarray]).form
    )
    assert (
        arrayt7.to_typetracer()._mergemany([emptyarray]).form
        == arrayt7._mergemany([emptyarray]).form
    )

    assert (
        emptyarray.to_typetracer()._mergemany([arrayt1]).form
        == emptyarray._mergemany([arrayt1]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayt2]).form
        == emptyarray._mergemany([arrayt2]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayt3]).form
        == emptyarray._mergemany([arrayt3]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayt4]).form
        == emptyarray._mergemany([arrayt4]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayt5]).form
        == emptyarray._mergemany([arrayt5]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayt6]).form
        == emptyarray._mergemany([arrayt6]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([arrayt7]).form
        == emptyarray._mergemany([arrayt7]).form
    )


def test_indexedarray_merge():
    content1 = ak.operations.from_iter(
        [[1.1, 2.2, 3.3], [], [4.4, 5.5]], highlevel=False
    )
    content2 = ak.operations.from_iter([[1, 2], [], [3, 4]], highlevel=False)
    index1 = ak.index.Index64(np.array([2, 0, -1, 0, 1, 2], dtype=np.int64))
    indexedarray1 = ak.contents.IndexedOptionArray(index1, content1)

    assert to_list(indexedarray1) == [
        [4.4, 5.5],
        [1.1, 2.2, 3.3],
        None,
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]

    assert to_list(indexedarray1._mergemany([content2])) == [
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
    assert to_list(content2._mergemany([indexedarray1])) == [
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
    assert to_list(indexedarray1._mergemany([indexedarray1])) == [
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

    assert (
        indexedarray1.to_typetracer()._mergemany([content2]).form
        == indexedarray1._mergemany([content2]).form
    )
    assert (
        content2.to_typetracer()._mergemany([indexedarray1]).form
        == content2._mergemany([indexedarray1]).form
    )
    assert (
        indexedarray1.to_typetracer()._mergemany([indexedarray1]).form
        == indexedarray1._mergemany([indexedarray1]).form
    )


def test_unionarray_merge():
    emptyarray = ak.contents.EmptyArray()

    one = ak.operations.from_iter([0.0, 1.1, 2.2, [], [1], [2, 2]], highlevel=False)
    two = ak.operations.from_iter(
        [{"x": 1, "y": 1.1}, 999, 123, {"x": 2, "y": 2.2}], highlevel=False
    )
    three = ak.operations.from_iter(["one", "two", "three"], highlevel=False)

    assert to_list(one._mergemany([two])) == [
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
    assert to_list(two._mergemany([one])) == [
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

    assert to_list(one._mergemany([emptyarray])) == [0.0, 1.1, 2.2, [], [1], [2, 2]]
    assert to_list(emptyarray._mergemany([one])) == [0.0, 1.1, 2.2, [], [1], [2, 2]]

    assert to_list(one._mergemany([three])) == [
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
    assert to_list(two._mergemany([three])) == [
        {"x": 1, "y": 1.1},
        999,
        123,
        {"x": 2, "y": 2.2},
        "one",
        "two",
        "three",
    ]
    assert to_list(three._mergemany([one])) == [
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
    assert to_list(three._mergemany([two])) == [
        "one",
        "two",
        "three",
        {"x": 1, "y": 1.1},
        999,
        123,
        {"x": 2, "y": 2.2},
    ]

    assert one.to_typetracer()._mergemany([two]).form == one._mergemany([two]).form
    assert two.to_typetracer()._mergemany([one]).form == two._mergemany([one]).form
    assert (
        one.to_typetracer()._mergemany([emptyarray]).form
        == one._mergemany([emptyarray]).form
    )
    assert (
        emptyarray.to_typetracer()._mergemany([one]).form
        == emptyarray._mergemany([one]).form
    )
    assert one.to_typetracer()._mergemany([three]).form == one._mergemany([three]).form
    assert two.to_typetracer()._mergemany([three]).form == two._mergemany([three]).form
    assert three.to_typetracer()._mergemany([one]).form == three._mergemany([one]).form
    assert three.to_typetracer()._mergemany([two]).form == three._mergemany([two]).form


def test_merge_parameters():
    one = ak.operations.from_iter(
        [[121, 117, 99, 107, 121], [115, 116, 117, 102, 102]], highlevel=False
    )
    two = ak.operations.from_iter(["good", "stuff"], highlevel=False)

    assert to_list(ak.operations.concatenate([one, two])) == [
        [121, 117, 99, 107, 121],
        [115, 116, 117, 102, 102],
        "good",
        "stuff",
    ]
    assert to_list(ak.operations.concatenate([two, one])) == [
        "good",
        "stuff",
        [121, 117, 99, 107, 121],
        [115, 116, 117, 102, 102],
    ]

    assert (
        ak.operations.concatenate([one, two], highlevel=False).to_typetracer().form
        == ak.operations.concatenate([one, two], highlevel=False).form
    )
    assert (
        ak.operations.concatenate([two, one], highlevel=False).to_typetracer().form
        == ak.operations.concatenate([two, one], highlevel=False).form
    )


def test_mask_as_bool():
    array = ak.operations.from_iter(
        ["one", "two", None, "three", None, None, "four"], highlevel=False
    )
    index2 = ak.index.Index64(np.array([2, 2, 1, 5, 0], dtype=np.int64))
    array2 = ak.contents.IndexedArray.simplified(index2, array)
    assert np.asarray(array.mask_as_bool(valid_when=False).view(np.int8)).tolist() == [
        0,
        0,
        1,
        0,
        1,
        1,
        0,
    ]
    assert np.asarray(array2.mask_as_bool(valid_when=False).view(np.int8)).tolist() == [
        1,
        1,
        0,
        1,
        0,
    ]


def test_indexedarray_simplify():
    array = ak.operations.from_iter(
        ["one", "two", None, "three", None, None, "four", "five"], highlevel=False
    )
    index2 = ak.index.Index64(np.array([2, 2, 1, 6, 5], dtype=np.int64))

    array2 = ak.contents.IndexedArray.simplified(index2, array)
    assert np.asarray(array.index).tolist() == [0, 1, -1, 2, -1, -1, 3, 4]
    assert to_list(array2) == to_list(array2) == [None, None, "two", "four", None]

    assert array2.to_typetracer().form == array2.form


def test_indexedarray_simplify_more():
    content = ak.contents.NumpyArray(
        np.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    )

    index1_32 = ak.index.Index32(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int32))
    index1_U32 = ak.index.IndexU32(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.uint32))
    index1_64 = ak.index.Index64(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int64))
    index2_32 = ak.index.Index32(np.array([0, 2, 4, 6], dtype=np.int32))
    index2_U32 = ak.index.IndexU32(np.array([0, 2, 4, 6], dtype=np.uint32))
    index2_64 = ak.index.Index64(np.array([0, 2, 4, 6], dtype=np.int64))

    array = ak.contents.IndexedArray.simplified(
        index2_32, ak.contents.IndexedArray(index1_32, content)
    )
    assert to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array, ak.contents.indexedarray.IndexedArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_32, ak.contents.IndexedArray(index1_U32, content)
    )
    assert to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array, ak.contents.indexedarray.IndexedArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_32, ak.contents.IndexedArray(index1_64, content)
    )
    assert to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array, ak.contents.indexedarray.IndexedArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_U32, ak.contents.IndexedArray(index1_32, content)
    )
    assert to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array, ak.contents.indexedarray.IndexedArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_U32, ak.contents.IndexedArray(index1_U32, content)
    )
    assert to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array, ak.contents.indexedarray.IndexedArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_U32, ak.contents.IndexedArray(index1_64, content)
    )
    assert to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array, ak.contents.indexedarray.IndexedArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_64, ak.contents.IndexedArray(index1_32, content)
    )
    assert to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array, ak.contents.indexedarray.IndexedArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_64, ak.contents.IndexedArray(index1_U32, content)
    )
    assert to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array, ak.contents.indexedarray.IndexedArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_64, ak.contents.IndexedArray(index1_64, content)
    )
    assert to_list(array) == [6.6, 4.4, 2.2, 0.0]
    assert isinstance(array, ak.contents.indexedarray.IndexedArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    index1_32 = ak.index.Index32(np.array([6, 5, -1, 3, -1, 1, 0], dtype=np.int32))
    index1_64 = ak.index.Index64(np.array([6, 5, -1, 3, -1, 1, 0], dtype=np.int64))
    index2_32 = ak.index.Index32(np.array([0, 2, 4, 6], dtype=np.int32))
    index2_U32 = ak.index.IndexU32(np.array([0, 2, 4, 6], dtype=np.uint32))
    index2_64 = ak.index.Index64(np.array([0, 2, 4, 6], dtype=np.int64))

    array = ak.contents.IndexedArray.simplified(
        index2_32, ak.contents.IndexedOptionArray(index1_32, content)
    )
    assert to_list(array) == [6.6, None, None, 0.0]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_32, ak.contents.IndexedOptionArray(index1_64, content)
    )
    assert to_list(array) == [6.6, None, None, 0.0]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_U32, ak.contents.IndexedOptionArray(index1_32, content)
    )
    assert to_list(array) == [6.6, None, None, 0.0]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_U32, ak.contents.IndexedOptionArray(index1_64, content)
    )
    assert to_list(array) == [6.6, None, None, 0.0]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_64, ak.contents.IndexedOptionArray(index1_32, content)
    )
    assert to_list(array) == [6.6, None, None, 0.0]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedArray.simplified(
        index2_64, ak.contents.IndexedOptionArray(index1_64, content)
    )
    assert to_list(array) == [6.6, None, None, 0.0]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    index1_32 = ak.index.Index32(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int32))
    index1_U32 = ak.index.IndexU32(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.uint32))
    index1_64 = ak.index.Index64(np.array([6, 5, 4, 3, 2, 1, 0], dtype=np.int64))
    index2_32 = ak.index.Index32(np.array([0, -1, 4, -1], dtype=np.int32))
    index2_64 = ak.index.Index64(np.array([0, -1, 4, -1], dtype=np.int64))

    array = ak.contents.IndexedOptionArray.simplified(
        index2_32, ak.contents.IndexedArray(index1_32, content)
    )
    assert to_list(array) == [6.6, None, 2.2, None]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedOptionArray.simplified(
        index2_32, ak.contents.IndexedArray(index1_U32, content)
    )
    assert to_list(array) == [6.6, None, 2.2, None]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedOptionArray.simplified(
        index2_32, ak.contents.IndexedArray(index1_64, content)
    )
    assert to_list(array) == [6.6, None, 2.2, None]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedOptionArray.simplified(
        index2_64, ak.contents.IndexedArray(index1_32, content)
    )
    assert to_list(array) == [6.6, None, 2.2, None]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedOptionArray.simplified(
        index2_64, ak.contents.IndexedArray(index1_U32, content)
    )
    assert to_list(array) == [6.6, None, 2.2, None]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedOptionArray.simplified(
        index2_64, ak.contents.IndexedArray(index1_64, content)
    )
    assert to_list(array) == [6.6, None, 2.2, None]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    index1_32 = ak.index.Index32(np.array([6, 5, -1, 3, -1, 1, 0], dtype=np.int32))
    index1_64 = ak.index.Index64(np.array([6, 5, -1, 3, -1, 1, 0], dtype=np.int64))
    index2_32 = ak.index.Index32(np.array([0, -1, 4, -1], dtype=np.int32))
    index2_64 = ak.index.Index64(np.array([0, -1, 4, -1], dtype=np.int64))

    array = ak.contents.IndexedOptionArray.simplified(
        index2_32, ak.contents.IndexedOptionArray(index1_32, content)
    )
    assert to_list(array) == [6.6, None, None, None]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedOptionArray.simplified(
        index2_32, ak.contents.IndexedOptionArray(index1_64, content)
    )
    assert to_list(array) == [6.6, None, None, None]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedOptionArray.simplified(
        index2_64, ak.contents.IndexedOptionArray(index1_32, content)
    )
    assert to_list(array) == [6.6, None, None, None]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form

    array = ak.contents.IndexedOptionArray.simplified(
        index2_64, ak.contents.IndexedOptionArray(index1_64, content)
    )
    assert to_list(array) == [6.6, None, None, None]
    assert isinstance(array, ak.contents.indexedoptionarray.IndexedOptionArray)
    assert isinstance(array.content, ak.contents.numpyarray.NumpyArray)
    assert array.to_typetracer().form == array.form


def test_unionarray_simplify_one():
    one = ak.operations.from_iter([5, 4, 3, 2, 1], highlevel=False)
    two = ak.operations.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    three = ak.operations.from_iter([1.1, 2.2, 3.3], highlevel=False)
    tags = ak.index.Index8(
        np.array([0, 0, 1, 2, 1, 0, 2, 1, 1, 0, 2, 0], dtype=np.int8)
    )
    index = ak.index.Index64(
        np.array([0, 1, 0, 0, 1, 2, 1, 2, 3, 3, 2, 4], dtype=np.int64)
    )
    array = ak.contents.UnionArray.simplified(tags, index, [one, two, three])

    assert to_list(array) == [
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
    assert len(array.contents) == 2
    assert array.to_typetracer().form == array.form


def test_unionarray_simplify():
    one = ak.operations.from_iter([5, 4, 3, 2, 1], highlevel=False)
    two = ak.operations.from_iter([[], [1], [2, 2], [3, 3, 3]], highlevel=False)
    three = ak.operations.from_iter([1.1, 2.2, 3.3], highlevel=False)

    tags2 = ak.index.Index8(np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int8))
    index2 = ak.index.Index32(np.array([0, 0, 1, 1, 2, 3, 2], dtype=np.int32))
    inner = ak.contents.UnionArray(tags2, index2, [two, three])
    tags1 = ak.index.Index8(
        np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0], dtype=np.int8)
    )
    index1 = ak.index.Index64(
        np.array([0, 1, 0, 1, 2, 2, 3, 4, 5, 3, 6, 4], dtype=np.int64)
    )
    outer = ak.contents.UnionArray.simplified(tags1, index1, [one, inner])
    assert to_list(outer) == [
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

    assert isinstance(outer.content(0), ak.contents.numpyarray.NumpyArray)
    assert isinstance(outer.content(1), ak.contents.listoffsetarray.ListOffsetArray)
    assert len(outer.contents) == 2
    assert outer.to_typetracer().form == outer.form

    tags2 = ak.index.Index8(np.array([0, 1, 0, 1, 0, 0, 1], dtype=np.int8))
    index2 = ak.index.Index64(np.array([0, 0, 1, 1, 2, 3, 2], dtype=np.int64))
    inner = ak.contents.UnionArray(tags2, index2, [two, three])
    tags1 = ak.index.Index8(
        np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1], dtype=np.int8)
    )
    index1 = ak.index.Index32(
        np.array([0, 1, 0, 1, 2, 2, 3, 4, 5, 3, 6, 4], dtype=np.int32)
    )
    outer = ak.contents.UnionArray.simplified(tags1, index1, [inner, one])
    assert to_list(outer) == [
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
    one = ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True).layout
    two = ak.highlevel.Array([[], [1], [2, 2], [3, 3, 3]], check_valid=True).layout
    three = ak.highlevel.Array(
        [True, False, False, True, True], check_valid=True
    ).layout

    assert to_list(ak.operations.concatenate([one, two, three])) == [
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
        ak.operations.concatenate([one, two, three], highlevel=False),
        ak.contents.unionarray.UnionArray,
    )
    assert (
        len(ak.operations.concatenate([one, two, three], highlevel=False).contents) == 2
    )


def test_where():
    condition = ak.highlevel.Array(
        [True, False, True, False, True],
        check_valid=True,
    )
    one = ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5], check_valid=True)
    two = ak.highlevel.Array([False, False, False, True, True], check_valid=True)
    three = ak.highlevel.Array(
        [[], [1], [2, 2], [3, 3, 3], [4, 4, 4, 4]], check_valid=True
    )

    assert to_list(ak.operations.where(condition, one, two)) == [
        1.1,
        0.0,
        3.3,
        1.0,
        5.5,
    ]
    assert to_list(ak.operations.where(condition, one, three)) == [
        1.1,
        [1],
        3.3,
        [3, 3, 3],
        5.5,
    ]

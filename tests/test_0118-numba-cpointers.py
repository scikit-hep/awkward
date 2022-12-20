# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

import operator
import sys

import numpy as np
import pytest

import awkward as ak

numba = pytest.importorskip("numba")

ak_numba_arrayview = pytest.importorskip("awkward._connect.numba.arrayview")

ak.numba.register_and_check()


def test_views():
    assert ak.operations.to_list(
        ak_numba_arrayview.ArrayView.fromarray(
            ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5])
        ).toarray()
    ) == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert ak.operations.to_list(
        ak_numba_arrayview.ArrayView.fromarray(
            ak.highlevel.Array(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))
        ).toarray()
    ) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    assert ak.operations.to_list(
        ak_numba_arrayview.ArrayView.fromarray(
            ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
        ).toarray()
    ) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    assert ak.operations.to_list(
        ak_numba_arrayview.ArrayView.fromarray(
            ak.highlevel.Array([1.1, 2.2, None, 3.3, None, 4.4, 5.5])
        ).toarray()
    ) == [1.1, 2.2, None, 3.3, None, 4.4, 5.5]

    assert ak.operations.to_list(
        ak_numba_arrayview.ArrayView.fromarray(
            ak.highlevel.Array(
                [
                    {"x": 0.0, "y": []},
                    {"x": 1.1, "y": [1, 1]},
                    {"x": 2.2, "y": [2, 2, 2]},
                ],
                check_valid=True,
            )
        ).toarray()
    ) == [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1, 1]}, {"x": 2.2, "y": [2, 2, 2]}]

    assert ak.operations.to_list(
        ak_numba_arrayview.ArrayView.fromarray(
            ak.highlevel.Array([(0.0, []), (1.1, [1, 1]), (2.2, [2, 2, 2])])
        ).toarray()
    ) == [(0.0, []), (1.1, [1, 1]), (2.2, [2, 2, 2])]

    assert ak.operations.to_list(
        ak_numba_arrayview.ArrayView.fromarray(
            ak.highlevel.Array([1.1, 2.2, 3.3, [], [1], [2, 2]])
        ).toarray()
    ) == [1.1, 2.2, 3.3, [], [1], [2, 2]]


def test_unbox():
    array = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    @numba.njit
    def f1(x):
        return 3.14

    assert f1(array) == 3.14


def test_box():
    array = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    @numba.njit
    def f1(x):
        return x

    assert ak.operations.to_list(f1(array)) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]


def test_refcount():
    array = ak.highlevel.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    array.numba_type
    assert [
        sys.getrefcount(x) == 2
        for x in (
            array._numbaview,
            array._numbaview.lookup,
            array._numbaview.lookup.positions,
            array._numbaview.lookup.arrayptrs,
        )
    ]

    for _ in range(3):

        @numba.njit
        def f1(x):
            return 3.14

        for _ in range(10):
            f1(array)
            assert [
                sys.getrefcount(x) == 2
                for x in (
                    array._numbaview,
                    array._numbaview.lookup,
                    array._numbaview.lookup.positions,
                    array._numbaview.lookup.arrayptrs,
                )
            ]

    for _ in range(3):

        @numba.njit
        def f2(x):
            return x

        for _ in range(10):
            y = f2(array)
            assert [
                sys.getrefcount(x) == 2
                for x in (
                    array._numbaview,
                    array._numbaview.lookup,
                    array._numbaview.lookup.positions,
                    array._numbaview.lookup.arrayptrs,
                )
            ]

    for _ in range(3):

        @numba.njit
        def f3(x):
            return x, x

        for _ in range(10):
            y = f3(array)  # noqa: F841 (checking reference count)
            assert [
                sys.getrefcount(x) == 2
                for x in (
                    array._numbaview,
                    array._numbaview.lookup,
                    array._numbaview.lookup.positions,
                    array._numbaview.lookup.arrayptrs,
                )
            ]


def test_len():
    array = ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    @numba.njit
    def f1(x):
        return len(x)

    assert f1(array) == 5


def test_NumpyArray_getitem():
    array = ak.highlevel.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    @numba.njit
    def f1(x, i):
        return x[i]

    assert f1(array, 0) == 1.1
    assert f1(array, 1) == 2.2
    assert f1(array, 2) == 3.3
    assert f1(array, 3) == 4.4
    assert f1(array, 4) == 5.5
    assert f1(array, -1) == 5.5
    assert f1(array, -2) == 4.4
    assert f1(array, -3) == 3.3
    assert f1(array, -4) == 2.2
    assert f1(array, -5) == 1.1

    with pytest.raises(ValueError) as err:
        assert f1(array, 5)
    assert str(err.value) == "slice index out of bounds"

    with pytest.raises(ValueError) as err:
        assert f1(array, -6)
    assert str(err.value) == "slice index out of bounds"

    @numba.njit
    def f2(x, i1, i2):
        return x[i1:i2]

    assert ak.operations.to_list(f2(array, 0, 5)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.to_list(f2(array, 1, 5)) == [2.2, 3.3, 4.4, 5.5]
    assert ak.operations.to_list(f2(array, 2, 5)) == [3.3, 4.4, 5.5]
    assert ak.operations.to_list(f2(array, 3, 5)) == [4.4, 5.5]
    assert ak.operations.to_list(f2(array, 4, 5)) == [5.5]
    assert ak.operations.to_list(f2(array, 5, 5)) == []
    assert ak.operations.to_list(f2(array, 6, 5)) == []
    assert ak.operations.to_list(f2(array, -1, 5)) == [5.5]
    assert ak.operations.to_list(f2(array, -2, 5)) == [4.4, 5.5]
    assert ak.operations.to_list(f2(array, -3, 5)) == [3.3, 4.4, 5.5]
    assert ak.operations.to_list(f2(array, -4, 5)) == [2.2, 3.3, 4.4, 5.5]
    assert ak.operations.to_list(f2(array, -5, 5)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.to_list(f2(array, -6, 5)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]

    assert ak.operations.to_list(f2(array, 0, -6)) == []
    assert ak.operations.to_list(f2(array, 0, -5)) == []
    assert ak.operations.to_list(f2(array, 0, -4)) == [1.1]
    assert ak.operations.to_list(f2(array, 0, -3)) == [1.1, 2.2]
    assert ak.operations.to_list(f2(array, 0, -2)) == [1.1, 2.2, 3.3]
    assert ak.operations.to_list(f2(array, 0, -1)) == [1.1, 2.2, 3.3, 4.4]
    assert ak.operations.to_list(f2(array, 0, 6)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.to_list(f2(array, 0, 5)) == [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
    ]
    assert ak.operations.to_list(f2(array, 0, 4)) == [1.1, 2.2, 3.3, 4.4]
    assert ak.operations.to_list(f2(array, 0, 3)) == [1.1, 2.2, 3.3]
    assert ak.operations.to_list(f2(array, 0, 2)) == [1.1, 2.2]
    assert ak.operations.to_list(f2(array, 0, 1)) == [1.1]
    assert ak.operations.to_list(f2(array, 0, 0)) == []

    aslist = [1.1, 2.2, 3.3, 4.4, 5.5]
    for i1 in range(-6, 7):
        for i2 in range(-6, 7):
            assert ak.operations.to_list(f2(array, i1, i2)) == aslist[i1:i2]

    @numba.njit
    def f3(x, i1, i2):
        return x[1:4][i1:i2]

    assert ak.operations.to_list(f3(array, -1, 3)) == [4.4]
    assert ak.operations.to_list(f3(array, -2, 3)) == [3.3, 4.4]
    assert ak.operations.to_list(f3(array, -3, 3)) == [2.2, 3.3, 4.4]
    assert ak.operations.to_list(f3(array, 0, 3)) == [2.2, 3.3, 4.4]
    assert ak.operations.to_list(f3(array, 1, 3)) == [3.3, 4.4]
    assert ak.operations.to_list(f3(array, 2, 3)) == [4.4]
    assert ak.operations.to_list(f3(array, 3, 3)) == []

    assert ak.operations.to_list(f3(array, 0, -4)) == []
    assert ak.operations.to_list(f3(array, 0, -3)) == []
    assert ak.operations.to_list(f3(array, 0, -2)) == [2.2]
    assert ak.operations.to_list(f3(array, 0, -1)) == [2.2, 3.3]
    assert ak.operations.to_list(f3(array, 0, 3)) == [2.2, 3.3, 4.4]
    assert ak.operations.to_list(f3(array, 0, 2)) == [2.2, 3.3]
    assert ak.operations.to_list(f3(array, 0, 1)) == [2.2]
    assert ak.operations.to_list(f3(array, 0, 0)) == []


def test_RegularArray_getitem():
    array = ak.highlevel.Array(np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))

    @numba.njit
    def f1(x, i):
        return x[i]

    assert f1(array, -2).to_list() == [1.1, 2.2, 3.3]
    assert f1(array, 0).to_list() == [1.1, 2.2, 3.3]
    assert f1(array, 1).to_list() == [4.4, 5.5, 6.6]
    assert f1(array, -1).to_list() == [4.4, 5.5, 6.6]

    @numba.njit
    def f2(x, i, j):
        return x[i][j]

    assert f2(array, 1, 0) == 4.4
    assert f2(array, 1, 1) == 5.5
    assert f2(array, 1, 2) == 6.6

    assert f2(array, -1, 0) == 4.4
    assert f2(array, -1, 1) == 5.5
    assert f2(array, -1, 2) == 6.6

    assert f2(array, 1, -3) == 4.4
    assert f2(array, 1, -2) == 5.5
    assert f2(array, 1, -1) == 6.6

    array = ak.highlevel.Array(np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]))

    @numba.njit
    def f3(x, i1, i2):
        return x[i1:i2]

    assert ak.operations.to_list(f3(array, -1, 3)) == [[5.5, 6.6]]
    assert ak.operations.to_list(f3(array, -2, 3)) == [
        [3.3, 4.4],
        [5.5, 6.6],
    ]
    assert ak.operations.to_list(f3(array, -3, 3)) == [
        [1.1, 2.2],
        [3.3, 4.4],
        [5.5, 6.6],
    ]
    assert ak.operations.to_list(f3(array, 0, 3)) == [
        [1.1, 2.2],
        [3.3, 4.4],
        [5.5, 6.6],
    ]
    assert ak.operations.to_list(f3(array, 1, 3)) == [
        [3.3, 4.4],
        [5.5, 6.6],
    ]
    assert ak.operations.to_list(f3(array, 2, 3)) == [[5.5, 6.6]]
    assert ak.operations.to_list(f3(array, 3, 3)) == []

    assert ak.operations.to_list(f3(array, 0, 0)) == []
    assert ak.operations.to_list(f3(array, 0, 1)) == [[1.1, 2.2]]
    assert ak.operations.to_list(f3(array, 0, 2)) == [
        [1.1, 2.2],
        [3.3, 4.4],
    ]
    assert ak.operations.to_list(f3(array, 0, 3)) == [
        [1.1, 2.2],
        [3.3, 4.4],
        [5.5, 6.6],
    ]
    assert ak.operations.to_list(f3(array, 0, -1)) == [
        [1.1, 2.2],
        [3.3, 4.4],
    ]
    assert ak.operations.to_list(f3(array, 0, -2)) == [[1.1, 2.2]]
    assert ak.operations.to_list(f3(array, 0, -3)) == []


def test_ListArray_getitem():
    array = ak.highlevel.Array(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    )

    @numba.njit
    def f1(x, i):
        return x[i]

    assert ak.operations.to_list(f1(array, 0)) == [0.0, 1.1, 2.2]
    assert ak.operations.to_list(f1(array, 1)) == []
    assert ak.operations.to_list(f1(array, 2)) == [3.3, 4.4]
    assert ak.operations.to_list(f1(array, 3)) == [5.5]
    assert ak.operations.to_list(f1(array, 4)) == [6.6, 7.7, 8.8, 9.9]

    @numba.njit
    def f2(x, i1, i2):
        return x[i1:i2]

    assert ak.operations.to_list(f2(array, 1, 4)) == [[], [3.3, 4.4], [5.5]]


def test_IndexedArray_getitem():
    content = ak.from_iter(
        [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9], highlevel=False
    )
    index = ak.index.Index64(np.array([3, 2, 2, 5, 0, 7], dtype=np.int64))
    array = ak.highlevel.Array(ak.contents.IndexedArray(index, content))

    @numba.njit
    def f1(x, i):
        return x[i]

    assert [f1(array, 0), f1(array, 1), f1(array, 2), f1(array, 3)] == [
        3.3,
        2.2,
        2.2,
        5.5,
    ]

    @numba.njit
    def f2(x, i1, i2):
        return x[i1:i2]

    assert ak.operations.to_list(f2(array, 1, 5)) == [2.2, 2.2, 5.5, 0]


def test_IndexedOptionArray_getitem():
    array = ak.highlevel.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])

    @numba.njit
    def f1(x, i):
        return x[i]

    assert [f1(array, 0), f1(array, 1), f1(array, 2), f1(array, 3)] == [
        1.1,
        2.2,
        None,
        3.3,
    ]

    @numba.njit
    def f2(x, i1, i2):
        return x[i1:i2]

    assert ak.operations.to_list(f2(array, 1, 5)) == [2.2, None, 3.3, None]


def test_RecordView_unbox_box():
    record = ak.highlevel.Array(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
            {"x": 4.4, "y": [4, 4, 4, 4]},
        ],
        check_valid=True,
    )[3]

    assert ak.operations.to_list(
        ak_numba_arrayview.RecordView.fromrecord(record).torecord()
    ) == {
        "x": 3.3,
        "y": [3, 3, 3],
    }

    @numba.njit
    def f1(x):
        return 3.14

    assert f1(record) == 3.14

    @numba.njit
    def f2(x):
        return x

    assert ak.operations.to_list(f2(record)) == {"x": 3.3, "y": [3, 3, 3]}


def test_RecordView_refcount():
    record = ak.highlevel.Array(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
            {"x": 4.4, "y": [4, 4, 4, 4]},
        ],
        check_valid=True,
    )[3]
    record.numba_type
    assert [
        sys.getrefcount(x) == 2
        for x in (
            record._numbaview,
            record._numbaview.arrayview,
            record._numbaview.arrayview.lookup,
            record._numbaview.arrayview.lookup.positions,
            record._numbaview.arrayview.lookup.arrayptrs,
        )
    ]

    for _ in range(3):

        @numba.njit
        def f1(x):
            return 3.14

        for _ in range(10):
            f1(record)
            assert [
                sys.getrefcount(x) == 2
                for x in (
                    record._numbaview,
                    record._numbaview.arrayview,
                    record._numbaview.arrayview.lookup,
                    record._numbaview.arrayview.lookup.positions,
                    record._numbaview.arrayview.lookup.arrayptrs,
                )
            ]

    for _ in range(3):

        @numba.njit
        def f2(x):
            return x

        for _ in range(10):
            y = f2(record)
            assert [
                sys.getrefcount(x) == 2
                for x in (
                    record._numbaview,
                    record._numbaview.arrayview,
                    record._numbaview.arrayview.lookup,
                    record._numbaview.arrayview.lookup.positions,
                    record._numbaview.arrayview.lookup.arrayptrs,
                )
            ]

    for _ in range(3):

        @numba.njit
        def f3(x):
            return x, x

        for _ in range(10):
            y = f3(record)  # noqa: F841 (checking reference count)
            assert [
                sys.getrefcount(x) == 2
                for x in (
                    record._numbaview,
                    record._numbaview.arrayview,
                    record._numbaview.arrayview.lookup,
                    record._numbaview.arrayview.lookup.positions,
                    record._numbaview.arrayview.lookup.arrayptrs,
                )
            ]


def test_Record_getitem():
    record = ak.highlevel.Array(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
            {"x": 4.4, "y": [4, 4, 4, 4]},
        ],
        check_valid=True,
    )[3]

    @numba.njit
    def f1(x):
        return x["x"]

    assert f1(record) == 3.3

    @numba.njit
    def f2(x):
        return x["y"]

    assert ak.operations.to_list(f2(record)) == [3, 3, 3]

    @numba.njit
    def f3(x):
        return x.x

    assert f3(record) == 3.3

    @numba.njit
    def f4(x):
        return x.y

    assert ak.operations.to_list(f4(record)) == [3, 3, 3]


def test_RecordArray_getitem():
    array = ak.highlevel.Array(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
            {"x": 4.4, "y": [4, 4, 4, 4]},
        ],
        check_valid=True,
    )

    @numba.njit
    def f1(x, i):
        return x[i]

    assert ak.operations.to_list(f1(array, 3)) == {"x": 3.3, "y": [3, 3, 3]}
    assert ak.operations.to_list(f1(array, 2)) == {"x": 2.2, "y": [2, 2]}
    assert ak.operations.to_list(f1(array, 1)) == {"x": 1.1, "y": [1]}

    @numba.njit
    def f2(x, i1, i2):
        return x[i1:i2]

    assert ak.operations.to_list(f2(array, 1, 4)) == [
        {"x": 1.1, "y": [1]},
        {"x": 2.2, "y": [2, 2]},
        {"x": 3.3, "y": [3, 3, 3]},
    ]

    array = ak.highlevel.Array(
        [
            [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
            [],
            [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}],
        ],
        check_valid=True,
    )

    @numba.njit
    def f3(x, i, j):
        return x[i][j]

    assert ak.operations.to_list(f3(array, 2, -2)) == {
        "x": 3.3,
        "y": [3, 3, 3],
    }


def test_RecordArray_getitem_field():
    array = ak.highlevel.Array(
        [
            {"x": 0.0, "y": []},
            {"x": 1.1, "y": [1]},
            {"x": 2.2, "y": [2, 2]},
            {"x": 3.3, "y": [3, 3, 3]},
            {"x": 4.4, "y": [4, 4, 4, 4]},
        ],
        check_valid=True,
    )

    @numba.njit
    def f1(x):
        return x[1:4]["x"]

    assert ak.operations.to_list(f1(array)) == [1.1, 2.2, 3.3]

    @numba.njit
    def f2(x):
        return x[1:4]["y"]

    assert ak.operations.to_list(f2(array)) == [[1], [2, 2], [3, 3, 3]]

    @numba.njit
    def f3(x):
        return x[1:4].x

    assert ak.operations.to_list(f3(array)) == [1.1, 2.2, 3.3]

    @numba.njit
    def f4(x):
        return x[1:4].y

    assert ak.operations.to_list(f4(array)) == [[1], [2, 2], [3, 3, 3]]


def test_ListArray_getitem_field():
    array = ak.highlevel.Array(
        [
            [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}],
            [],
            [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}],
            [{"x": 5.5, "y": [5, 5, 5, 5, 5]}],
            [
                {"x": 6.6, "y": [6, 6, 6, 6, 6, 6]},
                {"x": 7.7, "y": [7, 7, 7, 7, 7, 7, 7]},
                {"x": 8.8, "y": [8, 8, 8, 8, 8, 8, 8, 8]},
                {"x": 9.9, "y": [9, 9, 9, 9, 9, 9, 9, 9, 9]},
            ],
        ],
        check_valid=True,
    )

    @numba.njit
    def f1(x):
        return x["x"]

    assert ak.operations.to_list(f1(array)) == [
        [0.0, 1.1, 2.2],
        [],
        [3.3, 4.4],
        [5.5],
        [6.6, 7.7, 8.8, 9.9],
    ]

    @numba.njit
    def f2(x):
        return x.y

    assert ak.operations.to_list(f2(array)) == [
        [[], [1], [2, 2]],
        [],
        [[3, 3, 3], [4, 4, 4, 4]],
        [[5, 5, 5, 5, 5]],
        [
            [6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8],
            [9, 9, 9, 9, 9, 9, 9, 9, 9],
        ],
    ]

    @numba.njit
    def f3(x):
        return x[1:4].x

    assert ak.operations.to_list(f3(array)) == [[], [3.3, 4.4], [5.5]]

    @numba.njit
    def f4(x):
        return x[1:4]["y"]

    assert ak.operations.to_list(f4(array)) == [
        [],
        [[3, 3, 3], [4, 4, 4, 4]],
        [[5, 5, 5, 5, 5]],
    ]

    @numba.njit
    def f5(x):
        return x["x"][1:4]

    assert ak.operations.to_list(f5(array)) == [[], [3.3, 4.4], [5.5]]

    @numba.njit
    def f6(x):
        return x.y[1:4]

    assert ak.operations.to_list(f6(array)) == [
        [],
        [[3, 3, 3], [4, 4, 4, 4]],
        [[5, 5, 5, 5, 5]],
    ]

    @numba.njit
    def f7(x):
        return x[4]["x"]

    assert ak.operations.to_list(ak.operations.to_list(f7(array))) == [
        6.6,
        7.7,
        8.8,
        9.9,
    ]

    @numba.njit
    def f8(x):
        return x[4].y

    assert ak.operations.to_list(ak.operations.to_list(f8(array))) == [
        [6, 6, 6, 6, 6, 6],
        [7, 7, 7, 7, 7, 7, 7],
        [8, 8, 8, 8, 8, 8, 8, 8],
        [9, 9, 9, 9, 9, 9, 9, 9, 9],
    ]

    @numba.njit
    def f9(x):
        return x.x[4]

    assert ak.operations.to_list(ak.operations.to_list(f9(array))) == [
        6.6,
        7.7,
        8.8,
        9.9,
    ]

    @numba.njit
    def f10(x):
        return x["y"][4]

    assert ak.operations.to_list(ak.operations.to_list(f10(array))) == [
        [6, 6, 6, 6, 6, 6],
        [7, 7, 7, 7, 7, 7, 7],
        [8, 8, 8, 8, 8, 8, 8, 8],
        [9, 9, 9, 9, 9, 9, 9, 9, 9],
    ]

    @numba.njit
    def f11(x):
        return x[4]["x"][1]

    assert f11(array) == 7.7

    @numba.njit
    def f12(x):
        return x[4].y[1]

    assert ak.operations.to_list(ak.operations.to_list(f12(array))) == [
        7,
        7,
        7,
        7,
        7,
        7,
        7,
    ]

    @numba.njit
    def f12b(x):
        return x[4].y[1][6]

    assert ak.operations.to_list(ak.operations.to_list(f12b(array))) == 7

    @numba.njit
    def f13(x):
        return x.x[4][1]

    assert f13(array) == 7.7

    @numba.njit
    def f14(x):
        return x["y"][4][1]

    assert ak.operations.to_list(ak.operations.to_list(f14(array))) == [
        7,
        7,
        7,
        7,
        7,
        7,
        7,
    ]

    @numba.njit
    def f14b(x):
        return x["y"][4][1][6]

    assert ak.operations.to_list(f14b(array)) == 7


def test_RecordArray_deep_field():
    array = ak.highlevel.Array(
        [{"x": {"y": {"z": 1.1}}}, {"x": {"y": {"z": 2.2}}}, {"x": {"y": {"z": 3.3}}}],
        check_valid=True,
    )

    @numba.njit
    def f1(x):
        return x[1]["x"].y["z"]

    assert f1(array) == 2.2

    @numba.njit
    def f2(x):
        return x["x"][1].y["z"]

    assert f2(array) == 2.2

    @numba.njit
    def f3(x):
        return x["x"].y[1]["z"]

    assert f3(array) == 2.2

    @numba.njit
    def f4(x):
        return x["x"].y["z"][1]

    assert f4(array) == 2.2

    @numba.njit
    def f5(x):
        return x["x"].y["z"]

    assert ak.operations.to_list(f5(array)) == [1.1, 2.2, 3.3]

    @numba.njit
    def f6(x):
        return x.x["y"].z

    assert ak.operations.to_list(f6(array)) == [1.1, 2.2, 3.3]

    @numba.njit
    def f7(x):
        return x.x["y"]

    assert ak.operations.to_list(f7(array)) == [
        {"z": 1.1},
        {"z": 2.2},
        {"z": 3.3},
    ]

    @numba.njit
    def f8(x):
        return x.x

    assert ak.operations.to_list(f8(array)) == [
        {"y": {"z": 1.1}},
        {"y": {"z": 2.2}},
        {"y": {"z": 3.3}},
    ]


def test_ListArray_deep_at():
    content = ak.contents.NumpyArray(
        np.array(
            [
                1.1,
                2.2,
                3.3,
                4.4,
                5.5,
                6.6,
                7.7,
                8.8,
                9.9,
                10.0,
                11.1,
                12.2,
                13.3,
                14.4,
                15.5,
                16.6,
                17.7,
                18.8,
                19.9,
            ]
        )
    )
    offsets1 = ak.index.Index32(
        np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=np.int32)
    )
    listarray1 = ak.contents.ListOffsetArray(offsets1, content)
    offsets2 = ak.index.Index64(np.array([0, 2, 4, 6, 8], dtype=np.int64))
    listarray2 = ak.contents.ListOffsetArray(offsets2, listarray1)
    offsets3 = ak.index.Index32(np.array([0, 2, 4], dtype=np.int32))
    listarray3 = ak.contents.ListOffsetArray(offsets3, listarray2)
    array = ak.highlevel.Array(listarray3)

    @numba.njit
    def f1(x):
        return x[1][1][1][1]

    assert f1(array) == 16.6

    @numba.njit
    def f2(x):
        return x[1][1][1]

    assert ak.operations.to_list(f2(array)) == [15.5, 16.6]

    @numba.njit
    def f3(x):
        return x[1][1]

    assert ak.operations.to_list(f3(array)) == [[13.3, 14.4], [15.5, 16.6]]

    @numba.njit
    def f4(x):
        return x[1]

    assert ak.operations.to_list(f4(array)) == [
        [[9.9, 10.0], [11.1, 12.2]],
        [[13.3, 14.4], [15.5, 16.6]],
    ]


def test_IndexedArray_deep_at():
    content = ak.contents.NumpyArray(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    index1 = ak.index.Index32(np.array([1, 2, 3, 4], dtype=np.int32))
    indexedarray1 = ak.contents.IndexedArray(index1, content)
    index2 = ak.index.Index64(np.array([1, 2, 3], dtype=np.int64))
    indexedarray2 = ak.contents.IndexedArray.simplified(index2, indexedarray1)
    index3 = ak.index.Index32(np.array([1, 2], dtype=np.int32))
    indexedarray3 = ak.contents.IndexedArray.simplified(index3, indexedarray2)
    # This is not a valid array (IndexedArray inside IndexedArray), but instructive!  :)
    array = ak.highlevel.Array(indexedarray3, check_valid=False)

    @numba.njit
    def f1(x):
        return x[1]

    assert f1(array) == 5.5


def test_iterator():
    array = ak.highlevel.Array(
        [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    )

    @numba.njit
    def f1(a):
        out = 0.0
        for b in a:
            for c in b:
                out += c
        return out

    assert f1(array) == 49.5


def test_ArrayBuilder_refcount():
    builder = ak.highlevel.ArrayBuilder()
    assert (sys.getrefcount(builder), sys.getrefcount(builder._layout)) == (2, 2)

    @numba.njit
    def f1(x):
        return 3.14

    y = f1(builder)
    assert (sys.getrefcount(builder), sys.getrefcount(builder._layout)) == (2, 2)

    @numba.njit
    def f2(x):
        return x

    y = f2(builder)
    assert (sys.getrefcount(builder), sys.getrefcount(builder._layout)) == (2, 3)
    del y
    assert (sys.getrefcount(builder), sys.getrefcount(builder._layout)) == (2, 2)

    @numba.njit
    def f3(x):
        return x, x

    y = f3(builder)
    assert (sys.getrefcount(builder), sys.getrefcount(builder._layout)) == (2, 4)
    del y
    assert (sys.getrefcount(builder), sys.getrefcount(builder._layout)) == (2, 2)


def test_ArrayBuilder_len():
    builder = ak.highlevel.ArrayBuilder()
    builder.real(1.1)
    builder.real(2.2)
    builder.real(3.3)
    builder.real(4.4)
    builder.real(5.5)

    @numba.njit
    def f1(x):
        return len(x)

    assert f1(builder) == 5


def test_ArrayBuilder_simple():
    @numba.njit
    def f1(x):
        x.clear()
        return 3.14

    a = ak.highlevel.ArrayBuilder()
    f1(a)


def test_ArrayBuilder_boolean():
    @numba.njit
    def f1(x):
        x.boolean(True)
        x.boolean(False)
        x.boolean(False)
        return x

    a = ak.highlevel.ArrayBuilder()
    b = f1(a)
    assert ak.operations.to_list(a.snapshot()) == [True, False, False]
    assert ak.operations.to_list(b.snapshot()) == [True, False, False]


def test_ArrayBuilder_integer():
    @numba.njit
    def f1(x):
        x.integer(1)
        x.integer(2)
        x.integer(3)
        return x

    a = ak.highlevel.ArrayBuilder()
    b = f1(a)
    assert ak.operations.to_list(a.snapshot()) == [1, 2, 3]
    assert ak.operations.to_list(b.snapshot()) == [1, 2, 3]


def test_ArrayBuilder_real():
    @numba.njit
    def f1(x, z):
        x.real(1)
        x.real(2.2)
        x.real(z)
        return x

    a = ak.highlevel.ArrayBuilder()
    b = f1(a, np.array([3.5], dtype=np.float32)[0])
    assert ak.operations.to_list(a.snapshot()) == [1, 2.2, 3.5]
    assert ak.operations.to_list(b.snapshot()) == [1, 2.2, 3.5]


def test_ArrayBuilder_list():
    @numba.njit
    def f1(x):
        x.begin_list()
        x.real(1.1)
        x.real(2.2)
        x.real(3.3)
        x.end_list()
        x.begin_list()
        x.end_list()
        x.begin_list()
        x.real(4.4)
        x.real(5.5)
        x.end_list()
        return x

    a = ak.highlevel.ArrayBuilder()
    b = f1(a)
    assert ak.operations.to_list(a.snapshot()) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]
    assert ak.operations.to_list(b.snapshot()) == [
        [1.1, 2.2, 3.3],
        [],
        [4.4, 5.5],
    ]

    @numba.njit
    def f2(x):
        return len(x)

    assert f2(a) == 3
    assert f2(b) == 3

    @numba.njit
    def f3(x):
        x.clear()
        return x

    c = f3(b)
    assert ak.operations.to_list(a.snapshot()) == []
    assert ak.operations.to_list(b.snapshot()) == []
    assert ak.operations.to_list(c.snapshot()) == []


def test_ArrayBuilder_tuple():
    @numba.njit
    def f1(x):
        x.begin_tuple(2)
        x.index(0).append(1)
        x.index(1).append(1.1)
        x.end_tuple()
        x.begin_tuple(2)
        x.index(0).append(2)
        x.index(1).append(2.2)
        x.end_tuple()
        return x

    a = ak.highlevel.ArrayBuilder()
    b = f1(a)
    assert ak.operations.to_list(a.snapshot()) == [(1, 1.1), (2, 2.2)]
    assert ak.operations.to_list(b.snapshot()) == [(1, 1.1), (2, 2.2)]
    c = f1.py_func(a)
    assert ak.operations.to_list(a.snapshot()) == [
        (1, 1.1),
        (2, 2.2),
        (1, 1.1),
        (2, 2.2),
    ]
    assert ak.operations.to_list(c.snapshot()) == [
        (1, 1.1),
        (2, 2.2),
        (1, 1.1),
        (2, 2.2),
    ]


def test_ArrayBuilder_record():
    @numba.njit
    def f1(x):
        x.begin_record()
        x.field("x").append(1)
        x.field("y").append(1.1)
        x.end_record()
        x.begin_record()
        x.field("x").append(2)
        x.field("y").append(2.2)
        x.end_record()
        return x

    a = ak.highlevel.ArrayBuilder()
    b = f1(a)
    assert ak.operations.to_list(a.snapshot()) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
    ]
    assert ak.operations.to_list(b.snapshot()) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
    ]
    c = f1.py_func(a)
    assert ak.operations.to_list(a.snapshot()) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
    ]
    assert ak.operations.to_list(c.snapshot()) == [
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
        {"x": 1, "y": 1.1},
        {"x": 2, "y": 2.2},
    ]


def dummy_typer(viewtype):
    return numba.float64


def dummy_lower(context, builder, sig, args):
    def convert(rec):
        return rec.x + rec.y

    return context.compile_internal(builder, convert, sig, args)


def test_custom_record():
    behavior = {}
    behavior["__numba_typer__", "Dummy"] = dummy_typer
    behavior["__numba_lower__", "Dummy"] = dummy_lower

    array = ak.highlevel.Array(
        [{"x": 1.1, "y": 100}, {"x": 2.2, "y": 200}, {"x": 3.3, "y": 300}],
        behavior=behavior,
        check_valid=True,
    )
    array.layout.parameters["__record__"] = "Dummy"

    @numba.njit
    def f1(x, i):
        return x[i]

    assert f1(array, 1) == 202.2
    assert f1(array, 2) == 303.3


def dummy_typer2(viewtype):
    return numba.float64


def dummy_lower2(context, builder, sig, args):
    def compute(rec):
        return rec.x + rec.y

    return context.compile_internal(builder, compute, sig, args)


def test_custom_record2():
    behavior = {}
    behavior["__numba_typer__", "Dummy", "stuff"] = dummy_typer2
    behavior["__numba_lower__", "Dummy", "stuff"] = dummy_lower2

    array = ak.highlevel.Array(
        [{"x": 1.1, "y": 100}, {"x": 2.2, "y": 200}, {"x": 3.3, "y": 300}],
        behavior=behavior,
        check_valid=True,
    )
    array.layout.parameters["__record__"] = "Dummy"

    @numba.njit
    def f1(x, i):
        return x[i].stuff

    assert f1(array, 1) == 202.2
    assert f1(array, 2) == 303.3


def dummy_typer3(viewtype, args):
    return numba.float64(*args)


def dummy_lower3(context, builder, sig, args):
    def compute(rec, j):
        return rec.x + rec.y + j

    return context.compile_internal(builder, compute, sig, args)


def test_custom_record3():
    behavior = {}
    behavior["__numba_typer__", "Dummy", "stuff", ()] = dummy_typer3
    behavior["__numba_lower__", "Dummy", "stuff", ()] = dummy_lower3

    array = ak.highlevel.Array(
        [{"x": 1.1, "y": 100}, {"x": 2.2, "y": 200}, {"x": 3.3, "y": 300}],
        behavior=behavior,
        check_valid=True,
    )
    array.layout.parameters["__record__"] = "Dummy"

    @numba.njit
    def f1(x, i, j):
        return x[i].stuff(j)

    assert f1(array, 1, 1000) == 1202.2
    assert f1(array, 2, 1000) == 1303.3


def dummy_typer4(binop, left, right):
    return numba.float64(left, right)


def dummy_lower4(context, builder, sig, args):
    def compute(left, right):
        return left.x + right.x

    return context.compile_internal(builder, compute, sig, args)


def test_custom_record4():
    behavior = {}
    behavior["__numba_typer__", "Dummy", operator.add, "Dummy"] = dummy_typer4
    behavior["__numba_lower__", "Dummy", operator.add, "Dummy"] = dummy_lower4
    behavior["__numba_typer__", "Dummy", operator.eq, "Dummy"] = dummy_typer4
    behavior["__numba_lower__", "Dummy", operator.eq, "Dummy"] = dummy_lower4

    array = ak.highlevel.Array(
        [{"x": 1.1, "y": 100}, {"x": 2.2, "y": 200}, {"x": 3.3, "y": 300}],
        behavior=behavior,
        check_valid=True,
    )
    array.layout.parameters["__record__"] = "Dummy"

    @numba.njit
    def f1(x, i, j):
        return x[i] + x[j]

    assert f1(array, 1, 2) == 5.5

    @numba.njit
    def f2(x, i, j):
        return x[i] == x[j]

    assert f2(array, 1, 2) == 5.5


def dummy_typer5(unaryop, left):
    return numba.float64(left)


def dummy_lower5(context, builder, sig, args):
    def compute(left):
        return abs(left.x)

    return context.compile_internal(builder, compute, sig, args)


def test_custom_record5():
    behavior = {}
    behavior["__numba_typer__", abs, "Dummy"] = dummy_typer5
    behavior["__numba_lower__", abs, "Dummy"] = dummy_lower5
    behavior["__numba_typer__", operator.neg, "Dummy"] = dummy_typer5
    behavior["__numba_lower__", operator.neg, "Dummy"] = dummy_lower5

    array = ak.highlevel.Array(
        [{"x": 1.1, "y": 100}, {"x": -2.2, "y": 200}, {"x": 3.3, "y": 300}],
        behavior=behavior,
        check_valid=True,
    )
    array.layout.parameters["__record__"] = "Dummy"

    @numba.njit
    def f1(x, i):
        return abs(x[i])

    assert f1(array, 1) == 2.2
    assert f1(array, 2) == 3.3

    @numba.njit
    def f2(x, i):
        return -x[i]

    assert f2(array, 1) == 2.2
    assert f2(array, 2) == 3.3


def test_ArrayBuilder_append_numba5():
    @numba.njit
    def f1(builder, x):
        builder.append(x)

    @numba.njit
    def f2(builder, i):
        if i % 2 == 0:
            return 3
        else:
            return None

    @numba.njit
    def f3(builder, i):
        builder.append(f2(builder, i))

    builder = ak.highlevel.ArrayBuilder()

    f1(builder, True)
    f1(builder, 1)
    f1(builder, 2.2)
    f3(builder, 0)
    f3(builder, 1)
    f1(builder, None)

    assert ak.operations.to_list(builder.snapshot()) == [
        True,
        1,
        2.2,
        3,
        None,
        None,
    ]

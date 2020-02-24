# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1
import awkward1._numba.arrayview

numba = pytest.importorskip("numba")

def test_views():
    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5])).toarray()) == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array(numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))).toarray()) == [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])).toarray()) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([1.1, 2.2, None, 3.3, None, 4.4, 5.5])).toarray()) == [1.1, 2.2, None, 3.3, None, 4.4, 5.5]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1, 1]}, {"x": 2.2, "y": [2, 2, 2]}])).toarray()) == [{"x": 0.0, "y": []}, {"x": 1.1, "y": [1, 1]}, {"x": 2.2, "y": [2, 2, 2]}]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([(0.0, []), (1.1, [1, 1]), (2.2, [2, 2, 2])])).toarray()) == [(0.0, []), (1.1, [1, 1]), (2.2, [2, 2, 2])]

    assert awkward1.tolist(awkward1._numba.arrayview.ArrayView.fromarray(awkward1.Array([1.1, 2.2, 3.3, [], [1], [2, 2]])).toarray()) == [1.1, 2.2, 3.3, [], [1], [2, 2]]

def test_unbox():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    @numba.njit
    def f1(x):
        return 3.14

    assert f1(array) == 3.14

def test_box():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    @numba.njit
    def f1(x):
        return x

    assert awkward1.tolist(f1(array)) == [[1.1, 2.2, 3.3], [], [4.4, 5.5]]

def test_refcount():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])
    array.numbatype
    assert [sys.getrefcount(x) == 2 for x in (array._numbaview, array._numbaview.lookup, array._numbaview.lookup.positions, array._numbaview.lookup.arrayptrs)]

    for i in range(3):
        @numba.njit
        def f1(x):
            return 3.14
        
        for j in range(10):
            f1(array)
            assert [sys.getrefcount(x) == 2 for x in (array._numbaview, array._numbaview.lookup, array._numbaview.lookup.positions, array._numbaview.lookup.arrayptrs)]

    for i in range(3):
        @numba.njit
        def f2(x):
            return x
        
        for j in range(10):
            y = f2(array)
            assert [sys.getrefcount(x) == 2 for x in (array._numbaview, array._numbaview.lookup, array._numbaview.lookup.positions, array._numbaview.lookup.arrayptrs)]

    for i in range(3):
        @numba.njit
        def f3(x):
            return x, x

        for j in range(10):
            y = f3(array)
            assert [sys.getrefcount(x) == 2 for x in (array._numbaview, array._numbaview.lookup, array._numbaview.lookup.positions, array._numbaview.lookup.arrayptrs)]

def test_len():
    array = awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5])

    @numba.njit
    def f1(x):
        return len(x)

    assert f1(array) == 5

def test_NumpyArray_getitem():
    array = awkward1.Array([1.1, 2.2, 3.3, 4.4, 5.5])

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

    assert awkward1.tolist(f2(array,  0, 5)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array,  1, 5)) == [     2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array,  2, 5)) == [          3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array,  3, 5)) == [               4.4, 5.5]
    assert awkward1.tolist(f2(array,  4, 5)) == [                    5.5]
    assert awkward1.tolist(f2(array,  5, 5)) == [                       ]
    assert awkward1.tolist(f2(array,  6, 5)) == [                       ]
    assert awkward1.tolist(f2(array, -1, 5)) == [                    5.5]
    assert awkward1.tolist(f2(array, -2, 5)) == [               4.4, 5.5]
    assert awkward1.tolist(f2(array, -3, 5)) == [          3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array, -4, 5)) == [     2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array, -5, 5)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array, -6, 5)) == [1.1, 2.2, 3.3, 4.4, 5.5]

    assert awkward1.tolist(f2(array, 0, -6)) == [                       ]
    assert awkward1.tolist(f2(array, 0, -5)) == [                       ]
    assert awkward1.tolist(f2(array, 0, -4)) == [1.1                    ]
    assert awkward1.tolist(f2(array, 0, -3)) == [1.1, 2.2               ]
    assert awkward1.tolist(f2(array, 0, -2)) == [1.1, 2.2, 3.3          ]
    assert awkward1.tolist(f2(array, 0, -1)) == [1.1, 2.2, 3.3, 4.4     ]
    assert awkward1.tolist(f2(array, 0,  6)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array, 0,  5)) == [1.1, 2.2, 3.3, 4.4, 5.5]
    assert awkward1.tolist(f2(array, 0,  4)) == [1.1, 2.2, 3.3, 4.4     ]
    assert awkward1.tolist(f2(array, 0,  3)) == [1.1, 2.2, 3.3          ]
    assert awkward1.tolist(f2(array, 0,  2)) == [1.1, 2.2               ]
    assert awkward1.tolist(f2(array, 0,  1)) == [1.1                    ]
    assert awkward1.tolist(f2(array, 0,  0)) == [                       ]

    aslist = [1.1, 2.2, 3.3, 4.4, 5.5]
    for i1 in range(-6, 7):
        for i2 in range(-6, 7):
            assert awkward1.tolist(f2(array, i1, i2)) == aslist[i1:i2]

    @numba.njit
    def f3(x, i1, i2):
        return x[1:4][i1:i2]

    assert awkward1.tolist(f3(array, -1, 3)) == [          4.4]
    assert awkward1.tolist(f3(array, -2, 3)) == [     3.3, 4.4]
    assert awkward1.tolist(f3(array, -3, 3)) == [2.2, 3.3, 4.4]
    assert awkward1.tolist(f3(array,  0, 3)) == [2.2, 3.3, 4.4]
    assert awkward1.tolist(f3(array,  1, 3)) == [     3.3, 4.4]
    assert awkward1.tolist(f3(array,  2, 3)) == [          4.4]
    assert awkward1.tolist(f3(array,  3, 3)) == [             ]

    assert awkward1.tolist(f3(array, 0, -4)) == [             ]
    assert awkward1.tolist(f3(array, 0, -3)) == [             ]
    assert awkward1.tolist(f3(array, 0, -2)) == [2.2          ]
    assert awkward1.tolist(f3(array, 0, -1)) == [2.2, 3.3     ]
    assert awkward1.tolist(f3(array, 0,  3)) == [2.2, 3.3, 4.4]
    assert awkward1.tolist(f3(array, 0,  2)) == [2.2, 3.3     ]
    assert awkward1.tolist(f3(array, 0,  1)) == [2.2          ]
    assert awkward1.tolist(f3(array, 0,  0)) == [             ]

def test_RegularArray_getitem():
    array = awkward1.Array(numpy.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]))

    @numba.njit
    def f1(x, i):
        return x[i]

    assert f1(array, -2) == [1.1, 2.2, 3.3]
    assert f1(array,  0) == [1.1, 2.2, 3.3]
    assert f1(array,  1) == [4.4, 5.5, 6.6]
    assert f1(array, -1) == [4.4, 5.5, 6.6]

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

    array = awkward1.Array(numpy.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]))

    @numba.njit
    def f3(x, i1, i2):
        return x[i1:i2]

    assert awkward1.tolist(f3(array, -1, 3)) == [                        [5.5, 6.6]]
    assert awkward1.tolist(f3(array, -2, 3)) == [            [3.3, 4.4], [5.5, 6.6]]
    assert awkward1.tolist(f3(array, -3, 3)) == [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]
    assert awkward1.tolist(f3(array,  0, 3)) == [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]
    assert awkward1.tolist(f3(array,  1, 3)) == [            [3.3, 4.4], [5.5, 6.6]]
    assert awkward1.tolist(f3(array,  2, 3)) == [                        [5.5, 6.6]]
    assert awkward1.tolist(f3(array,  3, 3)) == [                                  ]

    assert awkward1.tolist(f3(array, 0,  0)) == [                                  ]
    assert awkward1.tolist(f3(array, 0,  1)) == [[1.1, 2.2]                        ]
    assert awkward1.tolist(f3(array, 0,  2)) == [[1.1, 2.2], [3.3, 4.4]            ]
    assert awkward1.tolist(f3(array, 0,  3)) == [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]
    assert awkward1.tolist(f3(array, 0, -1)) == [[1.1, 2.2], [3.3, 4.4]            ]
    assert awkward1.tolist(f3(array, 0, -2)) == [[1.1, 2.2]                        ]
    assert awkward1.tolist(f3(array, 0, -3)) == [                                  ]

def test_ListArray_getitem():
    array = awkward1.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])

    @numba.njit
    def f1(x, i):
        return x[i]

    assert awkward1.tolist(f1(array, 0)) == [0.0, 1.1, 2.2]
    assert awkward1.tolist(f1(array, 1)) == []
    assert awkward1.tolist(f1(array, 2)) == [3.3, 4.4]
    assert awkward1.tolist(f1(array, 3)) == [5.5]
    assert awkward1.tolist(f1(array, 4)) == [6.6, 7.7, 8.8, 9.9]

    @numba.njit
    def f2(x, i1, i2):
        return x[i1:i2]

    assert awkward1.tolist(f2(array, 1, 4)) == [[], [3.3, 4.4], [5.5]]

def test_IndexedArray_getitem():
    content = awkward1.Array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]).layout
    index = awkward1.layout.Index64(numpy.array([3, 2, 2, 5, 0, 7], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.IndexedArray64(index, content))

    @numba.njit
    def f1(x, i):
        return x[i]

    assert [f1(array, 0), f1(array, 1), f1(array, 2), f1(array, 3)] == [3.3, 2.2, 2.2, 5.5]

    @numba.njit
    def f2(x, i1, i2):
        return x[i1:i2]

    assert awkward1.tolist(f2(array, 1, 5)) == [2.2, 2.2, 5.5, 0]

def test_IndexedOptionArray_getitem():
    array = awkward1.Array([1.1, 2.2, None, 3.3, None, None, 4.4, 5.5])

    @numba.njit
    def f1(x, i):
        return x[i]

    assert [f1(array, 0), f1(array, 1), f1(array, 2), f1(array, 3)] == [1.1, 2.2, None, 3.3]

    @numba.njit
    def f2(x, i1, i2):
        return x[i1:i2]

    assert awkward1.tolist(f2(array, 1, 5)) == [2.2, None, 3.3, None]

def test_RecordView_unbox_box():
    record = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}])[3]

    assert awkward1.tolist(awkward1._numba.arrayview.RecordView.fromrecord(record).torecord()) == {"x": 3.3, "y": [3, 3, 3]}

    @numba.njit
    def f1(x):
        return 3.14

    assert f1(record) == 3.14

    @numba.njit
    def f2(x):
        return x

    assert awkward1.tolist(f2(record)) == {"x": 3.3, "y": [3, 3, 3]}

def test_RecordView_refcount():
    record = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}])[3]
    record.numbatype
    assert [sys.getrefcount(x) == 2 for x in (record._numbaview, record._numbaview.arrayview, record._numbaview.arrayview.lookup, record._numbaview.arrayview.lookup.positions, record._numbaview.arrayview.lookup.arrayptrs)]

    for i in range(3):
        @numba.njit
        def f1(x):
            return 3.14
        
        for j in range(10):
            f1(record)
            assert [sys.getrefcount(x) == 2 for x in (record._numbaview, record._numbaview.arrayview, record._numbaview.arrayview.lookup, record._numbaview.arrayview.lookup.positions, record._numbaview.arrayview.lookup.arrayptrs)]

    for i in range(3):
        @numba.njit
        def f2(x):
            return x
        
        for j in range(10):
            y = f2(record)
            assert [sys.getrefcount(x) == 2 for x in (record._numbaview, record._numbaview.arrayview, record._numbaview.arrayview.lookup, record._numbaview.arrayview.lookup.positions, record._numbaview.arrayview.lookup.arrayptrs)]

    for i in range(3):
        @numba.njit
        def f3(x):
            return x, x

        for j in range(10):
            y = f3(record)
            assert [sys.getrefcount(x) == 2 for x in (record._numbaview, record._numbaview.arrayview, record._numbaview.arrayview.lookup, record._numbaview.arrayview.lookup.positions, record._numbaview.arrayview.lookup.arrayptrs)]

def test_Record_getitem():
    record = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}])[3]
    @numba.njit
    def f1(x):
        return x["x"]

    assert f1(record) == 3.3

    @numba.njit
    def f2(x):
        return x["y"]

    assert awkward1.tolist(f2(record)) == [3, 3, 3]

    @numba.njit
    def f3(x):
        return x.x

    assert f3(record) == 3.3

    @numba.njit
    def f4(x):
        return x.y

    assert awkward1.tolist(f4(record)) == [3, 3, 3]

def test_RecordArray_getitem():
    array = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}])

    @numba.njit
    def f1(x, i):
        return x[i]

    assert awkward1.tolist(f1(array, 3)) == {"x": 3.3, "y": [3, 3, 3]}
    assert awkward1.tolist(f1(array, 2)) == {"x": 2.2, "y": [2, 2]}
    assert awkward1.tolist(f1(array, 1)) == {"x": 1.1, "y": [1]}

    @numba.njit
    def f2(x, i1, i2):
        return x[i1:i2]

    assert awkward1.tolist(f2(array, 1, 4)) == [{"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}]

    array = awkward1.Array([[{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}], [], [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]])

    @numba.njit
    def f3(x, i, j):
        return x[i][j]

    assert awkward1.tolist(f3(array, 2, -2)) == {"x": 3.3, "y": [3, 3, 3]}

def test_RecordArray_getitem_field():
    array = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}])

    @numba.njit
    def f1(x):
        return x[1:4]["x"]

    assert awkward1.tolist(f1(array)) == [1.1, 2.2, 3.3]

    @numba.njit
    def f2(x):
        return x[1:4]["y"]

    assert awkward1.tolist(f2(array)) == [[1], [2, 2], [3, 3, 3]]

    @numba.njit
    def f3(x):
        return x[1:4].x

    assert awkward1.tolist(f3(array)) == [1.1, 2.2, 3.3]

    @numba.njit
    def f4(x):
        return x[1:4].y

    assert awkward1.tolist(f4(array)) == [[1], [2, 2], [3, 3, 3]]

def test_ListArray_getitem_field():
    array = awkward1.Array([[{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}], [], [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}], [{"x": 5.5, "y": [5, 5, 5, 5, 5]}], [{"x": 6.6, "y": [6, 6, 6, 6, 6, 6]}, {"x": 7.7, "y": [7, 7, 7, 7, 7, 7, 7]}, {"x": 8.8, "y": [8, 8, 8, 8, 8, 8, 8, 8]}, {"x": 9.9, "y": [9, 9, 9, 9, 9, 9, 9, 9, 9]}]])

    @numba.njit
    def f1(x):
        return x["x"]

    assert awkward1.tolist(f1(array)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]

    @numba.njit
    def f2(x):
        return x.y

    assert awkward1.tolist(f2(array)) == [[[], [1], [2, 2]], [], [[3, 3, 3], [4, 4, 4, 4]], [[5, 5, 5, 5, 5]], [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9, 9, 9, 9]]]

    @numba.njit
    def f3(x):
        return x[1:4].x

    assert awkward1.tolist(f3(array)) == [[], [3.3, 4.4], [5.5]]

    @numba.njit
    def f4(x):
        return x[1:4]["y"]

    assert awkward1.tolist(f4(array)) == [[], [[3, 3, 3], [4, 4, 4, 4]], [[5, 5, 5, 5, 5]]]

    @numba.njit
    def f5(x):
        return x["x"][1:4]

    assert awkward1.tolist(f5(array)) == [[], [3.3, 4.4], [5.5]]

    @numba.njit
    def f6(x):
        return x.y[1:4]

    assert awkward1.tolist(f6(array)) == [[], [[3, 3, 3], [4, 4, 4, 4]], [[5, 5, 5, 5, 5]]]

    @numba.njit
    def f7(x):
        return x[4]["x"]

    assert awkward1.tolist(awkward1.tolist(f7(array))) == [6.6, 7.7, 8.8, 9.9]

    @numba.njit
    def f8(x):
        return x[4].y

    assert awkward1.tolist(awkward1.tolist(f8(array))) == [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9, 9, 9, 9]]

    @numba.njit
    def f9(x):
        return x.x[4]

    assert awkward1.tolist(awkward1.tolist(f9(array))) == [6.6, 7.7, 8.8, 9.9]

    @numba.njit
    def f10(x):
        return x["y"][4]

    assert awkward1.tolist(awkward1.tolist(f10(array))) == [[6, 6, 6, 6, 6, 6], [7, 7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9, 9, 9, 9]]

    @numba.njit
    def f11(x):
        return x[4]["x"][1]

    assert f11(array) == 7.7

    @numba.njit
    def f12(x):
        return x[4].y[1]

    assert awkward1.tolist(awkward1.tolist(f12(array))) == [7, 7, 7, 7, 7, 7, 7]

    @numba.njit
    def f12b(x):
        return x[4].y[1][6]

    assert awkward1.tolist(awkward1.tolist(f12b(array))) == 7

    @numba.njit
    def f13(x):
        return x.x[4][1]

    assert f13(array) == 7.7

    @numba.njit
    def f14(x):
        return x["y"][4][1]

    assert awkward1.tolist(awkward1.tolist(f14(array))) == [7, 7, 7, 7, 7, 7, 7]

    @numba.njit
    def f14b(x):
        return x["y"][4][1][6]

    assert awkward1.tolist(f14b(array)) == 7

def test_RecordArray_deep_field():
    array = awkward1.Array([{"x": {"y": {"z": 1.1}}}, {"x": {"y": {"z": 2.2}}}, {"x": {"y": {"z": 3.3}}}])

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

    assert awkward1.tolist(f5(array)) == [1.1, 2.2, 3.3]

    @numba.njit
    def f6(x):
        return x.x["y"].z

    assert awkward1.tolist(f6(array)) == [1.1, 2.2, 3.3]

    @numba.njit
    def f7(x):
        return x.x["y"]

    assert awkward1.tolist(f7(array)) == [{"z": 1.1}, {"z": 2.2}, {"z": 3.3}]

    @numba.njit
    def f8(x):
        return x.x

    assert awkward1.tolist(f8(array)) == [{"y": {"z": 1.1}}, {"y": {"z": 2.2}}, {"y": {"z": 3.3}}]

def test_ListArray_deep_at():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 12.2, 13.3, 14.4, 15.5, 16.6]))
    offsets1 = awkward1.layout.Index32(numpy.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], dtype=numpy.int32))
    listarray1 = awkward1.layout.ListOffsetArray32(offsets1, content)
    offsets2 = awkward1.layout.Index64(numpy.array([0, 2, 4, 6, 8], dtype=numpy.int64))
    listarray2 = awkward1.layout.ListOffsetArray64(offsets2, listarray1)
    offsets3 = awkward1.layout.Index32(numpy.array([0, 2, 4], dtype=numpy.int32))
    listarray3 = awkward1.layout.ListOffsetArray32(offsets3, listarray2)
    array = awkward1.Array(listarray3)

    @numba.njit
    def f1(x):
        return x[1][1][1][1]

    assert f1(array) == 16.6

    @numba.njit
    def f2(x):
        return x[1][1][1]

    assert awkward1.tolist(f2(array)) == [15.5, 16.6]

    @numba.njit
    def f3(x):
        return x[1][1]

    assert awkward1.tolist(f3(array)) == [[13.3, 14.4], [15.5, 16.6]]

    @numba.njit
    def f4(x):
        return x[1]

    assert awkward1.tolist(f4(array)) == [[[9.9, 10.0], [11.1, 12.2]], [[13.3, 14.4], [15.5, 16.6]]]

def test_IndexedArray_deep_at():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
    index1 = awkward1.layout.Index32(numpy.array([1, 2, 3, 4], dtype=numpy.int32))
    indexedarray1 = awkward1.layout.IndexedArray32(index1, content)
    index2 = awkward1.layout.Index64(numpy.array([1, 2, 3], dtype=numpy.int64))
    indexedarray2 = awkward1.layout.IndexedArray64(index2, indexedarray1)
    index3 = awkward1.layout.Index32(numpy.array([1, 2], dtype=numpy.int32))
    indexedarray3 = awkward1.layout.IndexedArray32(index3, indexedarray2)
    array = awkward1.Array(indexedarray3)

    @numba.njit
    def f1(x):
        return x[1]

    assert f1(array) == 5.5

def test_UnionArray_getitem():
    array = awkward1.Array([1, 2, 3, [], [1], [2, 2], {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}])

    content1 = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}]).layout
    content2 = awkward1.Array([{"y": [], "z": 0}, {"y": [1], "z": 1}, {"y": [2, 2], "z": 2}, {"y": [3, 3, 3], "z": 3}, {"y": [4, 4, 4, 4], "z": 4}]).layout
    tags  = awkward1.layout.Index8( numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [content1, content2]))

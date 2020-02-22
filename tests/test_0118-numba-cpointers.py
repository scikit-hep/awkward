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

def test_RecordArray_getitem():
    array = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}])

    # @numba.njit
    # def f1(x, i):
    #     return x[i]

    # print(f1(array, 1))



    # raise Exception


    # array = awkward1.Array([[{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}], [], [{"x": 3.3, "y": [3, 3, 3]}, {"x": 4.4, "y": [4, 4, 4, 4]}]])
    
def test_UnionArray_getitem():
    array = awkward1.Array([1, 2, 3, [], [1], [2, 2]])

    content1 = awkward1.Array([{"x": 0.0, "y": []}, {"x": 1.1, "y": [1]}, {"x": 2.2, "y": [2, 2]}, {"x": 3.3, "y": [3, 3, 3]}]).layout
    content2 = awkward1.Array([{"y": [], "z": 0}, {"y": [1], "z": 1}, {"y": [2, 2], "z": 2}, {"y": [3, 3, 3], "z": 3}, {"y": [4, 4, 4, 4], "z": 4}]).layout
    tags  = awkward1.layout.Index8( numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=numpy.int8))
    index = awkward1.layout.Index64(numpy.array([0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=numpy.int64))
    array = awkward1.Array(awkward1.layout.UnionArray8_64(tags, index, [content1, content2]))

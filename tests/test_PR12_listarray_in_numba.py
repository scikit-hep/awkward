# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1
awkward1_numba_util = pytest.importorskip("awkward1._numba.util")

py27 = (sys.version_info[0] < 3)

def test_slice_utils():
    a = numpy.array([1, 2, 3])
    b = numpy.array([[4], [5], [6], [7]])
    c = 999

    assert [x.tolist() for x in numpy.broadcast_arrays(a, b)] == [x.tolist() for x in awkward1_numba_util.broadcast_arrays((a, b))]
    assert [x.tolist() for x in numpy.broadcast_arrays(b, c)] == [x.tolist() for x in awkward1_numba_util.broadcast_arrays((b, c))]
    assert [x.tolist() for x in numpy.broadcast_arrays(c, a)] == [x.tolist() for x in awkward1_numba_util.broadcast_arrays((c, a))]
    assert [x.tolist() for x in numpy.broadcast_arrays(a, b, c)] == [x.tolist() for x in awkward1_numba_util.broadcast_arrays((a, b, c))]

    if not py27:
        assert awkward1_numba_util.broadcast_arrays(("hello", a))[0] == "hello"
        assert awkward1_numba_util.broadcast_arrays((a, "hello"))[1] == "hello"
        assert awkward1_numba_util.broadcast_arrays(("hello", c)) == ("hello", c)
        assert awkward1_numba_util.broadcast_arrays((c, "hello")) == (c, "hello")
        assert [x.tolist() for x in numpy.broadcast_arrays(a, b)] == [x.tolist() for x in awkward1_numba_util.broadcast_arrays(("hello", a, b))[1:]]

    assert [x.tolist() for x in awkward1_numba_util.maskarrays_to_indexarrays((numpy.array([0, 1, 2, 3]), numpy.array([[True, False], [False, True], [True, True], [False, False]])))]

starts  = numpy.array([0, 3, 3, 5, 6])
stops   = numpy.array([3, 3, 5, 6, 10])
content = numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
array   = awkward1.layout.ListArray64(awkward1.layout.Index64(starts),
                                      awkward1.layout.Index64(stops),
                                      awkward1.layout.NumpyArray(content))

def test_boxing():
    assert awkward1.tolist(array) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]
    @numba.njit
    def f1(q):
        return q
    assert awkward1.tolist(f1(array)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]

def test_simple():
    @numba.njit
    def f1(q):
        return q[2]

    assert awkward1.tolist(f1(array)) == [3.3, 4.4]

    @numba.njit
    def f2(q):
        return q[2:4]

    assert awkward1.tolist(f2(array)) == [[3.3, 4.4], [5.5]]

    @numba.njit
    def f3(q):
        return q[()]

    assert awkward1.tolist(f3(array)) == [[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]]

#     @numba.njit
#     def f2(q):
#         return q[numpy.array([2, 0, 0, 1]),]
#
#     print(f2(array))

    # raise Exception

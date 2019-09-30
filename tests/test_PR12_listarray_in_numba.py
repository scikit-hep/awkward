# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1
awkward1_numba_content = pytest.importorskip("awkward1._numba.content")

py27 = (sys.version_info[0] < 3)

def test_slice_utils():
    a = numpy.array([1, 2, 3])
    b = numpy.array([[4], [5], [6], [7]])
    c = 999

    assert [x.tolist() for x in numpy.broadcast_arrays(a, b)] == [x.tolist() for x in awkward1_numba_content.broadcast_arrays((a, b))]
    assert [x.tolist() for x in numpy.broadcast_arrays(b, c)] == [x.tolist() for x in awkward1_numba_content.broadcast_arrays((b, c))]
    assert [x.tolist() for x in numpy.broadcast_arrays(c, a)] == [x.tolist() for x in awkward1_numba_content.broadcast_arrays((c, a))]
    assert [x.tolist() for x in numpy.broadcast_arrays(a, b, c)] == [x.tolist() for x in awkward1_numba_content.broadcast_arrays((a, b, c))]

    assert awkward1_numba_content.broadcast_arrays(("hello", a))[0] == "hello"
    assert awkward1_numba_content.broadcast_arrays((a, "hello"))[1] == "hello"
    assert awkward1_numba_content.broadcast_arrays(("hello", c)) == ("hello", c)
    assert awkward1_numba_content.broadcast_arrays((c, "hello")) == (c, "hello")

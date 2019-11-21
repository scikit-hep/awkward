# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

import pytest
import numpy

import awkward1

def test_basic():
    content1 = awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4, 5]))
    content2 = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 9]))
    listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content2)
    recordarray = awkward1.layout.RecordArray()
    recordarray.append(content1, "one")
    recordarray.append(listoffsetarray, "two")
    recordarray.append(content2)
    recordarray.setkey(0, "wonky")
    assert awkward1.tolist(recordarray.field(0)) == [1, 2, 3, 4, 5]
    assert awkward1.tolist(recordarray.field("two")) == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6], [7.7, 8.8, 9.9]]
    assert awkward1.tolist(recordarray.field("wonky")) == [1, 2, 3, 4, 5]
    str(recordarray)
    assert awkward1.tojson(recordarray) == '[{"wonky":1,"two":[1.1,2.2,3.3],"2":1.1},{"wonky":2,"two":[],"2":2.2},{"wonky":3,"two":[4.4,5.5],"2":3.3},{"wonky":4,"two":[6.6],"2":4.4},{"wonky":5,"two":[7.7,8.8,9.9],"2":5.5}]'

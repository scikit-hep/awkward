# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

def test_record():
    array1 = awkward1.from_iter([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], highlevel=False)
    assert awkward1.to_list(array1) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]

    array2 = array1.setitem_field("z", awkward1.from_iter([[], [1], [2, 2]], highlevel=False))
    assert awkward1.to_list(array2) == [{"x": 1, "y": 1.1, "z": []}, {"x": 2, "y": 2.2, "z": [1]}, {"x": 3, "y": 3.3, "z": [2, 2]}]

    array3 = array1.setitem_field(None, awkward1.from_iter([[], [1], [2, 2]], highlevel=False))
    assert awkward1.to_list(array3) == [{"x": 1, "y": 1.1, "2": []}, {"x": 2, "y": 2.2, "2": [1]}, {"x": 3, "y": 3.3, "2": [2, 2]}]

    array3 = array1.setitem_field(0, awkward1.from_iter([[], [1], [2, 2]], highlevel=False))
    assert awkward1.to_list(array3) == [{"x": 1, "y": 1.1, "0": []}, {"x": 2, "y": 2.2, "0": [1]}, {"x": 3, "y": 3.3, "0": [2, 2]}]

    array1 = awkward1.from_iter([(1, 1.1), (2, 2.2), (3, 3.3)], highlevel=False)
    assert awkward1.to_list(array1) == [(1, 1.1), (2, 2.2), (3, 3.3)]

    array2 = array1.setitem_field("z", awkward1.from_iter([[], [1], [2, 2]], highlevel=False))
    assert awkward1.to_list(array2) == [{"0": 1, "1": 1.1, "z": []}, {"0": 2, "1": 2.2, "z": [1]}, {"0": 3, "1": 3.3, "z": [2, 2]}]

    array3 = array1.setitem_field(None, awkward1.from_iter([[], [1], [2, 2]], highlevel=False))
    assert awkward1.to_list(array3) == [(1, 1.1, []), (2, 2.2, [1]), (3, 3.3, [2, 2])]

    array3 = array1.setitem_field(0, awkward1.from_iter([[], [1], [2, 2]], highlevel=False))
    assert awkward1.to_list(array3) == [([], 1, 1.1), ([1], 2, 2.2), ([2, 2], 3, 3.3)]

    array3 = array1.setitem_field(1, awkward1.from_iter([[], [1], [2, 2]], highlevel=False))
    assert awkward1.to_list(array3) == [(1, [], 1.1), (2, [1], 2.2), (3, [2, 2], 3.3)]

    array3 = array1.setitem_field(100, awkward1.from_iter([[], [1], [2, 2]], highlevel=False))
    assert awkward1.to_list(array3) == [(1, 1.1, []), (2, 2.2, [1]), (3, 3.3, [2, 2])]

def test_withfield():
    base = awkward1.Array([{"x": 1}, {"x": 2}, {"x": 3}], check_valid=True)
    what = awkward1.Array([1.1, 2.2, 3.3], check_valid=True)
    assert awkward1.to_list(awkward1.with_field(base, what)) == [{"x": 1, "1": 1.1}, {"x": 2, "1": 2.2}, {"x": 3, "1": 3.3}]
    assert awkward1.to_list(awkward1.with_field(base, what, where="y")) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]

    base["z"] = what
    assert awkward1.to_list(base) == [{"x": 1, "z": 1.1}, {"x": 2, "z": 2.2}, {"x": 3, "z": 3.3}]

    base["q"] = 123
    assert awkward1.to_list(base) == [{"x": 1, "z": 1.1, "q": 123}, {"x": 2, "z": 2.2, "q": 123}, {"x": 3, "z": 3.3, "q": 123}]

    base = awkward1.Array([{"x": 1}, {"x": 2}, {"x": 3}], check_valid=True)[2]
    assert awkward1.to_list(awkward1.with_field(base, 100, "y")) == {"x": 3, "y": 100}

def test_regulararray():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    recordarray = awkward1.layout.RecordArray({"x": content})
    regulararray = awkward1.Array(awkward1.layout.RegularArray(recordarray, 3), check_valid=True)

    content2 = awkward1.layout.NumpyArray(numpy.array([100, 200, 300]))
    regulararray2 = awkward1.Array(awkward1.layout.RegularArray(content2, 1), check_valid=True)
    assert awkward1.to_list(awkward1.with_field(regulararray, regulararray2, "y")) == [[{"x": 0.0, "y": 100}, {"x": 1.1, "y": 100}, {"x": 2.2, "y": 100}], [{"x": 3.3, "y": 200}, {"x": 4.4, "y": 200}, {"x": 5.5, "y": 200}], [{"x": 6.6, "y": 300}, {"x": 7.7, "y": 300}, {"x": 8.8, "y": 300}]]

    content2 = awkward1.layout.NumpyArray(numpy.array([100, 200, 300, 400, 500, 600, 700, 800, 900]))
    regulararray2 = awkward1.Array(awkward1.layout.RegularArray(content2, 3), check_valid=True)
    assert awkward1.to_list(awkward1.with_field(regulararray, regulararray2, "y")) == [[{"x": 0.0, "y": 100}, {"x": 1.1, "y": 200}, {"x": 2.2, "y": 300}], [{"x": 3.3, "y": 400}, {"x": 4.4, "y": 500}, {"x": 5.5, "y": 600}], [{"x": 6.6, "y": 700}, {"x": 7.7, "y": 800}, {"x": 8.8, "y": 900}]]

    content2 = awkward1.Array(awkward1.layout.NumpyArray(numpy.array([[100], [200], [300]])), check_valid=True)
    assert awkward1.to_list(awkward1.with_field(regulararray, content2, "y")) == [[{"x": 0.0, "y": 100}, {"x": 1.1, "y": 100}, {"x": 2.2, "y": 100}], [{"x": 3.3, "y": 200}, {"x": 4.4, "y": 200}, {"x": 5.5, "y": 200}], [{"x": 6.6, "y": 300}, {"x": 7.7, "y": 300}, {"x": 8.8, "y": 300}]]

    content2 = awkward1.Array(awkward1.layout.NumpyArray(numpy.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]])), check_valid=True)
    assert awkward1.to_list(awkward1.with_field(regulararray, content2, "y")) == [[{"x": 0.0, "y": 100}, {"x": 1.1, "y": 200}, {"x": 2.2, "y": 300}], [{"x": 3.3, "y": 400}, {"x": 4.4, "y": 500}, {"x": 5.5, "y": 600}], [{"x": 6.6, "y": 700}, {"x": 7.7, "y": 800}, {"x": 8.8, "y": 900}]]

def test_listarray():
    one = awkward1.Array([[{"x": 1}, {"x": 2}, {"x": 3}], [], [{"x": 4}, {"x": 5}]], check_valid=True)
    two = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]], check_valid=True)
    assert awkward1.to_list(awkward1.with_field(one, two, "y")) == [[{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}], [], [{"x": 4, "y": 4.4}, {"x": 5, "y": 5.5}]]

    three = awkward1.Array([100, 200, 300], check_valid=True)
    assert awkward1.to_list(awkward1.with_field(one, three, "y")) == [[{"x": 1, "y": 100}, {"x": 2, "y": 100}, {"x": 3, "y": 100}], [], [{"x": 4, "y": 300}, {"x": 5, "y": 300}]]

    assert awkward1.to_list(awkward1.with_field(one, [100, 200, 300], "y")) == [[{"x": 1, "y": 100}, {"x": 2, "y": 100}, {"x": 3, "y": 100}], [], [{"x": 4, "y": 300}, {"x": 5, "y": 300}]]

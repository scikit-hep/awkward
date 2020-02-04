# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

def test_record():
    array1 = awkward1.Array([{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]).layout
    assert awkward1.tolist(array1) == [{"x": 1, "y": 1.1}, {"x": 2, "y": 2.2}, {"x": 3, "y": 3.3}]

    array2 = array1.setitem_field("z", awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array2) == [{"x": 1, "y": 1.1, "z": []}, {"x": 2, "y": 2.2, "z": [1]}, {"x": 3, "y": 3.3, "z": [2, 2]}]

    array3 = array1.setitem_field(None, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [{"x": 1, "y": 1.1, "2": []}, {"x": 2, "y": 2.2, "2": [1]}, {"x": 3, "y": 3.3, "2": [2, 2]}]

    array3 = array1.setitem_field(0, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [{"x": 1, "y": 1.1, "0": []}, {"x": 2, "y": 2.2, "0": [1]}, {"x": 3, "y": 3.3, "0": [2, 2]}]

    array1 = awkward1.Array([(1, 1.1), (2, 2.2), (3, 3.3)]).layout
    assert awkward1.tolist(array1) == [(1, 1.1), (2, 2.2), (3, 3.3)]

    array2 = array1.setitem_field("z", awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array2) == [{"0": 1, "1": 1.1, "z": []}, {"0": 2, "1": 2.2, "z": [1]}, {"0": 3, "1": 3.3, "z": [2, 2]}]

    array3 = array1.setitem_field(None, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [(1, 1.1, []), (2, 2.2, [1]), (3, 3.3, [2, 2])]

    array3 = array1.setitem_field(0, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [([], 1, 1.1), ([1], 2, 2.2), ([2, 2], 3, 3.3)]

    array3 = array1.setitem_field(1, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [(1, [], 1.1), (2, [1], 2.2), (3, [2, 2], 3.3)]

    array3 = array1.setitem_field(100, awkward1.Array([[], [1], [2, 2]]).layout)
    assert awkward1.tolist(array3) == [(1, 1.1, []), (2, 2.2, [1]), (3, 3.3, [2, 2])]

def test_regulararray():
    content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]))
    recordarray = awkward1.layout.RecordArray({"x": content})
    regulararray = awkward1.layout.RegularArray(recordarray, 3)

    content2 = awkward1.layout.NumpyArray(numpy.array([100, 200, 300]))
    regulararray2 = awkward1.layout.RegularArray(content2, 1)
    assert awkward1.tolist(regulararray.setitem_field("y", regulararray2)) == [[{"x": 0.0, "y": 100}, {"x": 1.1, "y": 100}, {"x": 2.2, "y": 100}], [{"x": 3.3, "y": 200}, {"x": 4.4, "y": 200}, {"x": 5.5, "y": 200}], [{"x": 6.6, "y": 300}, {"x": 7.7, "y": 300}, {"x": 8.8, "y": 300}]]

    content2 = awkward1.layout.NumpyArray(numpy.array([100, 200, 300, 400, 500, 600, 700, 800, 900]))
    regulararray2 = awkward1.layout.RegularArray(content2, 3)
    assert awkward1.tolist(regulararray.setitem_field("y", regulararray2)) == [[{"x": 0.0, "y": 100}, {"x": 1.1, "y": 200}, {"x": 2.2, "y": 300}], [{"x": 3.3, "y": 400}, {"x": 4.4, "y": 500}, {"x": 5.5, "y": 600}], [{"x": 6.6, "y": 700}, {"x": 7.7, "y": 800}, {"x": 8.8, "y": 900}]]

    content2 = awkward1.layout.NumpyArray(numpy.array([[100], [200], [300]]))
    assert awkward1.tolist(regulararray.setitem_field("y", content2)) == [[{"x": 0.0, "y": 100}, {"x": 1.1, "y": 100}, {"x": 2.2, "y": 100}], [{"x": 3.3, "y": 200}, {"x": 4.4, "y": 200}, {"x": 5.5, "y": 200}], [{"x": 6.6, "y": 300}, {"x": 7.7, "y": 300}, {"x": 8.8, "y": 300}]]

    content2 = awkward1.layout.NumpyArray(numpy.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]]))
    assert awkward1.tolist(regulararray.setitem_field("y", content2)) == [[{"x": 0.0, "y": 100}, {"x": 1.1, "y": 200}, {"x": 2.2, "y": 300}], [{"x": 3.3, "y": 400}, {"x": 4.4, "y": 500}, {"x": 5.5, "y": 600}], [{"x": 6.6, "y": 700}, {"x": 7.7, "y": 800}, {"x": 8.8, "y": 900}]]

    content2 = awkward1.Array([[100], [200], [300]]).layout
    assert awkward1.tolist(regulararray.setitem_field("y", content2)) == [[{"x": 0.0, "y": 100}, {"x": 1.1, "y": 100}, {"x": 2.2, "y": 100}], [{"x": 3.3, "y": 200}, {"x": 4.4, "y": 200}, {"x": 5.5, "y": 200}], [{"x": 6.6, "y": 300}, {"x": 7.7, "y": 300}, {"x": 8.8, "y": 300}]]

    content2 = awkward1.Array([[100, 200, 300], [400, 500, 600], [700, 800, 900]]).layout
    assert awkward1.tolist(regulararray.setitem_field("y", content2)) == [[{"x": 0.0, "y": 100}, {"x": 1.1, "y": 200}, {"x": 2.2, "y": 300}], [{"x": 3.3, "y": 400}, {"x": 4.4, "y": 500}, {"x": 5.5, "y": 600}], [{"x": 6.6, "y": 700}, {"x": 7.7, "y": 800}, {"x": 8.8, "y": 900}]]

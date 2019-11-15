# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import os
import json

import pytest
import numpy

import awkward1

content = awkward1.layout.NumpyArray(numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]));
offsets = awkward1.layout.Index64(numpy.array([0, 3, 3, 5, 6, 10, 10]))
listoffsetarray = awkward1.layout.ListOffsetArray64(offsets, content)
regulararray = awkward1.layout.RegularArray(listoffsetarray, 2)
starts = awkward1.layout.Index64(numpy.array([0, 1]))
stops = awkward1.layout.Index64(numpy.array([2, 3]))
listarray = awkward1.layout.ListArray64(starts, stops, regulararray)

def test_type():
    assert str(awkward1.typeof(regulararray)) == "3 * 2 * var * float64"

def test_iteration():
    assert awkward1.tolist(regulararray) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]

def test_getitem_at():
    assert awkward1.tolist(regulararray[0]) == [[0.0, 1.1, 2.2], []]
    assert awkward1.tolist(regulararray[1]) == [[3.3, 4.4], [5.5]]
    assert awkward1.tolist(regulararray[2]) == [[6.6, 7.7, 8.8, 9.9], []]

def test_getitem_range():
    assert awkward1.tolist(regulararray[1:]) == [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert awkward1.tolist(regulararray[:-1]) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]]

def test_getitem():
    assert awkward1.tolist(regulararray[(0,)]) == [[0.0, 1.1, 2.2], []]
    assert awkward1.tolist(regulararray[(1,)]) == [[3.3, 4.4], [5.5]]
    assert awkward1.tolist(regulararray[(2,)]) == [[6.6, 7.7, 8.8, 9.9], []]
    assert awkward1.tolist(regulararray[(slice(1, None, None),)]) == [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert awkward1.tolist(regulararray[(slice(None, -1, None),)]) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]]

def test_getitem_deeper():
    assert awkward1.tolist(listarray) == [[[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]], [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]]

    assert awkward1.tolist(listarray[0, 0, 0]) == [0.0, 1.1, 2.2]
    assert awkward1.tolist(listarray[0, 0, 1]) == []
    assert awkward1.tolist(listarray[0, 1, 0]) == [3.3, 4.4]
    assert awkward1.tolist(listarray[0, 1, 1]) == [5.5]
    assert awkward1.tolist(listarray[1, 0, 0]) == [3.3, 4.4]
    assert awkward1.tolist(listarray[1, 0, 1]) == [5.5]
    assert awkward1.tolist(listarray[1, 1, 0]) == [6.6, 7.7, 8.8, 9.9]
    assert awkward1.tolist(listarray[1, 1, 1]) == []

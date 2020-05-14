# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

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

def test_simple_type():
    assert str(awkward1.type(content)) == "float64"

def test_type():
    assert str(awkward1.type(regulararray)) == "2 * var * float64"

def test_iteration():
    assert awkward1.to_list(regulararray) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]

def test_tojson():
    assert awkward1.to_json(regulararray) == "[[[0.0,1.1,2.2],[]],[[3.3,4.4],[5.5]],[[6.6,7.7,8.8,9.9],[]]]"

def test_getitem_at():
    assert awkward1.to_list(regulararray[0]) == [[0.0, 1.1, 2.2], []]
    assert awkward1.to_list(regulararray[1]) == [[3.3, 4.4], [5.5]]
    assert awkward1.to_list(regulararray[2]) == [[6.6, 7.7, 8.8, 9.9], []]

def test_getitem_range():
    assert awkward1.to_list(regulararray[1:]) == [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert awkward1.to_list(regulararray[:-1]) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]]

def test_getitem():
    assert awkward1.to_list(regulararray[(0,)]) == [[0.0, 1.1, 2.2], []]
    assert awkward1.to_list(regulararray[(1,)]) == [[3.3, 4.4], [5.5]]
    assert awkward1.to_list(regulararray[(2,)]) == [[6.6, 7.7, 8.8, 9.9], []]
    assert awkward1.to_list(regulararray[(slice(1, None, None),)]) == [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]
    assert awkward1.to_list(regulararray[(slice(None, -1, None),)]) == [[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]]

def test_getitem_deeper():
    assert awkward1.to_list(listarray) == [[[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]], [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]]

    assert awkward1.to_list(listarray[0, 0, 0]) == [0.0, 1.1, 2.2]
    assert awkward1.to_list(listarray[0, 0, 1]) == []
    assert awkward1.to_list(listarray[0, 1, 0]) == [3.3, 4.4]
    assert awkward1.to_list(listarray[0, 1, 1]) == [5.5]
    assert awkward1.to_list(listarray[1, 0, 0]) == [3.3, 4.4]
    assert awkward1.to_list(listarray[1, 0, 1]) == [5.5]
    assert awkward1.to_list(listarray[1, 1, 0]) == [6.6, 7.7, 8.8, 9.9]
    assert awkward1.to_list(listarray[1, 1, 1]) == []

    assert awkward1.to_list(listarray[0, 0, 0:]) == [[0.0, 1.1, 2.2], []]
    assert awkward1.to_list(listarray[0, 0, 1:]) == [[]]
    assert awkward1.to_list(listarray[0, 1, 0:]) == [[3.3, 4.4], [5.5]]
    assert awkward1.to_list(listarray[0, 1, 1:]) == [[5.5]]
    assert awkward1.to_list(listarray[1, 0, 0:]) == [[3.3, 4.4], [5.5]]
    assert awkward1.to_list(listarray[1, 0, 1:]) == [[5.5]]
    assert awkward1.to_list(listarray[1, 1, 0:]) == [[6.6, 7.7, 8.8, 9.9], []]
    assert awkward1.to_list(listarray[1, 1, 1:]) == [[]]

    assert awkward1.to_list(listarray[[1], 0, 0:]) == [[[3.3, 4.4], [5.5]]]
    assert awkward1.to_list(listarray[[1, 0], 0, 0:]) == [[[3.3, 4.4], [5.5]], [[0.0, 1.1, 2.2], []]]

    assert awkward1.to_list(listarray[:, :, [0, 1]]) == [[[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]], [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]]
    assert awkward1.to_list(listarray[:, :, [1, 0]]) == [[[[], [0.0, 1.1, 2.2]], [[5.5], [3.3, 4.4]]], [[[5.5], [3.3, 4.4]], [[], [6.6, 7.7, 8.8, 9.9]]]]
    assert awkward1.to_list(listarray[:, :, [1, 0, 1]]) == [[[[], [0.0, 1.1, 2.2], []], [[5.5], [3.3, 4.4], [5.5]]], [[[5.5], [3.3, 4.4], [5.5]], [[], [6.6, 7.7, 8.8, 9.9], []]]]
    assert awkward1.to_list(listarray[:, :2, [0, 1]]) == [[[[0.0, 1.1, 2.2], []], [[3.3, 4.4], [5.5]]], [[[3.3, 4.4], [5.5]], [[6.6, 7.7, 8.8, 9.9], []]]]

    assert awkward1.to_list(listarray[:1, [0, 0, 1, 1], [0, 1, 0, 1]]) == [[[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5]]]
    assert awkward1.to_list(listarray[:1, [1, 1, 0, 0], [1, 0, 1, 0]]) == [[[5.5], [3.3, 4.4], [], [0.0, 1.1, 2.2]]]

content2 = awkward1.layout.NumpyArray(numpy.arange(2*3*5*7).reshape(-1, 7))
regulararrayA = awkward1.layout.RegularArray(content2, 5)
regulararrayB = awkward1.layout.RegularArray(regulararrayA, 3)
modelA = numpy.arange(2*3*5*7).reshape(2*3, 5, 7)
modelB = numpy.arange(2*3*5*7).reshape(2, 3, 5, 7)

def test_numpy():
    assert awkward1.to_list(regulararrayA) == awkward1.to_list(modelA)
    assert awkward1.to_list(regulararrayB) == awkward1.to_list(modelB)

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((0, 1, 4, -5), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(regulararrayA[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(None), slice(1, None), slice(None, -1), slice(None, None, 2)), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(regulararrayA[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(1, None), slice(None, -1), 2, -2), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(regulararrayA[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], 2, -2), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(regulararrayA[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], slice(1, None), slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(regulararrayA[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-2, -1, 0, 1, 1), depth):
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(regulararrayB[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-1, 0, 1, slice(1, None), slice(None, -1)), depth):
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(regulararrayB[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-1, 0, [1, 0, 0, 1], [0, 1, -1, 1], slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(regulararrayB[cuts])

def test_setidentities():
    regulararray.setidentities()
    assert numpy.asarray(regulararray.identities).tolist() == [
        [0],
        [1],
        [2]]
    assert numpy.asarray(regulararray.content.identities).tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1]]
    assert numpy.asarray(regulararray.content.content.identities).tolist() == [
        [0, 0, 0],   # 0.0
        [0, 0, 1],   # 1.1
        [0, 0, 2],   # 2.2
      # [0, 1,  ],   # (empty list)
        [1, 0, 0],   # 3.3
        [1, 0, 1],   # 4.4
        [1, 1, 0],   # 5.5
        [2, 0, 0],   # 6.6
        [2, 0, 1],   # 7.7
        [2, 0, 2],   # 8.8
        [2, 0, 3]]   # 9.9
      # [2, 1,  ],   # (empty list)

    regulararrayB.setidentities()
    assert numpy.asarray(regulararrayB.identities).tolist() == [
        [0],
        [1]]
    assert numpy.asarray(regulararrayB.content.identities).tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2]]

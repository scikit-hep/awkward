# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

content = awkward1.layout.NumpyArray(numpy.arange(2*3*5*7).reshape(-1, 7))
offsetsA = numpy.arange(0, 2*3*5 + 5, 5)
offsetsB = numpy.arange(0, 2*3 + 3, 3)
startsA, stopsA = offsetsA[:-1], offsetsA[1:]
startsB, stopsB = offsetsB[:-1], offsetsB[1:]

listoffsetarrayA64 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(offsetsA), content)
listoffsetarrayA32 = awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(offsetsA), content)
listarrayA64 = awkward1.layout.ListArray64(awkward1.layout.Index64(startsA), awkward1.layout.Index64(stopsA), content)
listarrayA32 = awkward1.layout.ListArray32(awkward1.layout.Index32(startsA), awkward1.layout.Index32(stopsA), content)
modelA = numpy.arange(2*3*5*7).reshape(2*3, 5, 7)

listoffsetarrayB64 = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(offsetsB), listoffsetarrayA64)
listoffsetarrayB32 = awkward1.layout.ListOffsetArray32(awkward1.layout.Index32(offsetsB), listoffsetarrayA64)
listarrayB64 = awkward1.layout.ListArray64(awkward1.layout.Index64(startsB), awkward1.layout.Index64(stopsB), listarrayA64)
listarrayB32 = awkward1.layout.ListArray32(awkward1.layout.Index32(startsB), awkward1.layout.Index32(stopsB), listarrayA64)
modelB = numpy.arange(2*3*5*7).reshape(2, 3, 5, 7)

listoffsetarrayB64.setidentities()
listoffsetarrayB32.setidentities()
listarrayB64.setidentities()
listarrayB32.setidentities()

def test_basic():
    assert awkward1.to_list(modelA) == awkward1.to_list(listoffsetarrayA64)
    assert awkward1.to_list(modelA) == awkward1.to_list(listoffsetarrayA32)
    assert awkward1.to_list(modelA) == awkward1.to_list(listarrayA64)
    assert awkward1.to_list(modelA) == awkward1.to_list(listarrayA32)
    assert awkward1.to_list(modelB) == awkward1.to_list(listoffsetarrayB64)
    assert awkward1.to_list(modelB) == awkward1.to_list(listoffsetarrayB32)
    assert awkward1.to_list(modelB) == awkward1.to_list(listarrayB64)
    assert awkward1.to_list(modelB) == awkward1.to_list(listarrayB32)

def test_listoffsetarrayA64():
    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((0, 1, 4, -5), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listoffsetarrayA64[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(None), slice(1, None), slice(None, -1), slice(None, None, 2)), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listoffsetarrayA64[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(1, None), slice(None, -1), 2, -2), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listoffsetarrayA64[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], 2, -2), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listoffsetarrayA64[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], slice(1, None), slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listoffsetarrayA64[cuts])

def test_listoffsetarrayA32():
    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((0, 1, 4, -5), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listoffsetarrayA32[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(None), slice(1, None), slice(None, -1), slice(None, None, 2)), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listoffsetarrayA32[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(1, None), slice(None, -1), 2, -2), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listoffsetarrayA32[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], 2, -2), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listoffsetarrayA32[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], slice(1, None), slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listoffsetarrayA32[cuts])

def test_listarrayA64():
    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((0, 1, 4, -5), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listarrayA64[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(None), slice(1, None), slice(None, -1), slice(None, None, 2)), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listarrayA64[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(1, None), slice(None, -1), 2, -2), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listarrayA64[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], 2, -2), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listarrayA64[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], slice(1, None), slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listarrayA64[cuts])

def test_listarrayA32():
    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((0, 1, 4, -5), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listarrayA32[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(None), slice(1, None), slice(None, -1), slice(None, None, 2)), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listarrayA32[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations((slice(1, None), slice(None, -1), 2, -2), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listarrayA32[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], 2, -2), depth):
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listarrayA32[cuts])

    for depth in 0, 1, 2, 3:
        for cuts in itertools.permutations(([2, 0, 0, 1], [1, -2, 0, -1], slice(1, None), slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.to_list(modelA[cuts]) == awkward1.to_list(listarrayA32[cuts])

def test_listoffsetarrayB64():
    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-2, -1, 0, 1, 1), depth):
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listoffsetarrayB64[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-1, 0, 1, slice(1, None), slice(None, -1)), depth):
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listoffsetarrayB64[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-1, 0, [1, 0, 0, 1], [0, 1, -1, 1], slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listoffsetarrayB64[cuts])

def test_listoffsetarrayB32():
    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-2, -1, 0, 1, 1), depth):
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listoffsetarrayB64[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-1, 0, 1, slice(1, None), slice(None, -1)), depth):
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listoffsetarrayB64[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-1, 0, [1, 0, 0, 1], [0, 1, -1, 1], slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listoffsetarrayB64[cuts])

def test_listarrayB64():
    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-2, -1, 0, 1, 1), depth):
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listarrayB64[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-1, 0, 1, slice(1, None), slice(None, -1)), depth):
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listarrayB64[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-1, 0, [1, 0, 0, 1], [0, 1, -1, 1], slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listarrayB64[cuts])

def test_listarrayB32():
    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-2, -1, 0, 1, 1), depth):
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listarrayB64[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-1, 0, 1, slice(1, None), slice(None, -1)), depth):
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listarrayB64[cuts])

    for depth in 0, 1, 2, 3, 4:
        for cuts in itertools.permutations((-1, 0, [1, 0, 0, 1], [0, 1, -1, 1], slice(None, -1)), depth):
            cuts = cuts
            while len(cuts) > 0 and isinstance(cuts[0], slice):
                cuts = cuts[1:]
            while len(cuts) > 0 and isinstance(cuts[-1], slice):
                cuts = cuts[:-1]
            if any(isinstance(x, slice) for x in cuts):
                continue
            assert awkward1.to_list(modelB[cuts]) == awkward1.to_list(listarrayB64[cuts])

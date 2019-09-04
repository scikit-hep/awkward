# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

def test_slice():
    print(awkward1.layout.testslice(3))
    print(awkward1.layout.testslice(slice(1, 2)))
    print(awkward1.layout.testslice(slice(None, 2)))
    print(awkward1.layout.testslice(slice(1, None)))
    print(awkward1.layout.testslice(slice(1, 2, 3)))
    print(awkward1.layout.testslice(slice(None, 2, 3)))
    print(awkward1.layout.testslice(slice(1, None, 3)))
    if not py27:
        print(awkward1.layout.testslice(Ellipsis))
    print(awkward1.layout.testslice(numpy.newaxis))
    print(awkward1.layout.testslice(None))
    print(awkward1.layout.testslice(numpy.array([1, 2, 3, 4, 5, 6, 7], dtype="i4")))
    print(awkward1.layout.testslice(numpy.array([1, 2, 3, 4, 5], dtype="i8")))
    print(awkward1.layout.testslice([1, 2, 3, 4, 5]))
    print(awkward1.layout.testslice(numpy.array([True, False, True])))
    print(awkward1.layout.testslice([True, False, True]))
    print(awkward1.layout.testslice((3, slice(1, 2), slice(None, None, 3), 0 if py27 else Ellipsis, numpy.newaxis, None, [1, 2, 3, 4, 5], numpy.array([1, 2, 3, 4, 5, 6, 7], dtype="i4"), [True, False, True])))
    print(awkward1.layout.testslice(()))
    print(awkward1.layout.testslice((3,)))
    assert False, "intentionally fail"

# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

py27 = (sys.version_info[0] < 3)

def test_slice():
    awkward1.layout.testslice(3)
    awkward1.layout.testslice(slice(1, 2))
    awkward1.layout.testslice(slice(None, 2))
    awkward1.layout.testslice(slice(1, None))
    awkward1.layout.testslice(slice(1, 2, 3))
    awkward1.layout.testslice(slice(None, 2, 3))
    awkward1.layout.testslice(slice(1, None, 3))
    if not py27:
        awkward1.layout.testslice(...)
    awkward1.layout.testslice(numpy.newaxis)
    awkward1.layout.testslice(None)
    awkward1.layout.testslice(numpy.array([1, 2, 3, 4, 5, 6, 7], dtype="i4"))
    awkward1.layout.testslice(numpy.array([1, 2, 3, 4, 5], dtype="i8"))
    awkward1.layout.testslice([1, 2, 3, 4, 5])
    awkward1.layout.testslice(numpy.array([True, False, True]))
    awkward1.layout.testslice([True, False, True])
    awkward1.layout.testslice((3, slice(1, 2), slice(None, None, 3), 0 if py27 else ..., numpy.newaxis, None, [1, 2, 3, 4, 5], numpy.array([1, 2, 3, 4, 5, 6, 7], dtype="i4"), [True, False, True]))
    awkward1.layout.testslice(())
    awkward1.layout.testslice((3,))

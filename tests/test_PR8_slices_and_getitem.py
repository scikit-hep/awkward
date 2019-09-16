# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import itertools

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
        awkward1.layout.testslice(Ellipsis)
    awkward1.layout.testslice(numpy.newaxis)
    awkward1.layout.testslice(None)
    awkward1.layout.testslice(numpy.array([1, 2, 3, 4, 5, 6, 7], dtype="i4"))
    awkward1.layout.testslice(numpy.array([1, 2, 3, 4, 5], dtype="i8"))
    awkward1.layout.testslice([1, 2, 3, 4, 5])
    awkward1.layout.testslice(numpy.array([True, False, True]))
    awkward1.layout.testslice([True, False, True])
    awkward1.layout.testslice((3, slice(1, 2), slice(None, None, 3), 0 if py27 else Ellipsis, numpy.newaxis, None, [1, 2, 3, 4, 5], numpy.array([1, 2, 3, 4, 5, 6, 7], dtype="i4"), [True, False, True]))
    awkward1.layout.testslice(())
    awkward1.layout.testslice((3,))
    with pytest.raises(ValueError):
        awkward1.layout.testslice(numpy.array([1.1, 2.2, 3.3]))
    with pytest.raises(ValueError):
        awkward1.layout.testslice(numpy.array(["one", "two", "three"]))
    with pytest.raises(ValueError):
        awkward1.layout.testslice(numpy.array([1, 2, 3, None, 4, 5]))

def test_numpyarray_getitem():
    a = numpy.arange(120).reshape(6, 4, 5)
    b = awkward1.layout.NumpyArray(a)

    for depth in 1, 2, 3:
        for cuts in itertools.permutations((0, 1, 2, slice(0, 2), slice(1, 3), slice(1, 4)), depth):
            if sum(1 if isinstance(x, slice) else 0 for x in cuts) <= 1:
                print(cuts)
                acut = awkward1.tolist(a[cuts])
                bcut = awkward1.tolist(b.getitem(cuts))
                print(acut)
                print(bcut)
                print()
                assert acut == bcut

    cuts = (slice(1, 3), slice(0, 2))
    acut = a[cuts]
    bcut = b.getitem(cuts)
    print(awkward1.tolist(acut), acut.shape)
    print(awkward1.tolist(bcut), bcut.shape)

    # raise Exception

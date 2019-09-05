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
    # a = numpy.array([0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9])
    # b = awkward1.layout.NumpyArray(a)
    # assert b.getitem(4) == a[4]
    # assert b.getitem(-4) == a[-4]
    # assert awkward1.tolist(b.getitem(slice(3, 8))) == awkward1.tolist(a[3:8])
    # assert awkward1.tolist(b.getitem(slice(-7, -2))) == awkward1.tolist(a[-7:-2])
    # assert awkward1.tolist(b.getitem(slice(3, None))) == awkward1.tolist(a[3:])
    # assert awkward1.tolist(b.getitem(slice(None, -2))) == awkward1.tolist(a[:-2])
    # assert awkward1.tolist(b.getitem(slice(None, None))) == awkward1.tolist(a[:])
    # assert awkward1.tolist(b.getitem(slice(3, 100))) == awkward1.tolist(a[3:100])
    # assert awkward1.tolist(b.getitem(slice(-100, -2))) == awkward1.tolist(a[-100:-2])

    a = numpy.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]], dtype="i2")
    b = awkward1.layout.NumpyArray(a)
    # assert b.getitem((1, 3)) == a[1, 3]
    # print(awkward1.tolist(b.getitem((2, slice(1, 4)))), awkward1.tolist(a[2, 1:4]))
    # print(awkward1.tolist(b.getitem((slice(1, 3), 2))), awkward1.tolist(a[1:3, 2]))
    # print(awkward1.tolist(a[1:3]))
    # print((b.getitem((slice(1, 3), 2))))

    print(awkward1.tolist(a[(slice(1, 3),)]))
    print(awkward1.tolist(b.getitem((slice(1, 3),))))
    assert awkward1.tolist(a[(slice(1, 3),)]) == awkward1.tolist(b.getitem((slice(1, 3),)))

    print(awkward1.tolist(a[(slice(1, 3), 2)]))
    print(awkward1.tolist(b.getitem((slice(1, 3), 2))))
    assert awkward1.tolist(a[(slice(1, 3), 2)]) == awkward1.tolist(b.getitem((slice(1, 3), 2)))

    print("----------------------------------------------------")
    print(awkward1.tolist(a[(2, slice(1, 3))]))
    print(awkward1.tolist(b.getitem((2, slice(1, 3)))))
    assert awkward1.tolist(a[(2, slice(1, 3))]) == awkward1.tolist(b.getitem((2, slice(1, 3))))

    raise Exception

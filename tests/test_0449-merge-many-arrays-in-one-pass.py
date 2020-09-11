# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy
import awkward1


def test_numpyarray():
    for dtype1 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
        for dtype2 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
            for dtype3 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
                for dtype4 in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f4", "f8", "?"):
                    one   = numpy.array([0, 1, 2], dtype=dtype1)
                    two   = numpy.array([3, 0], dtype=dtype2)
                    three = numpy.array([], dtype=dtype3)
                    four  = numpy.array([4, 5, 0, 6, 7], dtype=dtype4)
                    combined = numpy.concatenate([one, two, three, four])

                    ak_combined = awkward1.layout.NumpyArray(one).mergemany([
                        awkward1.layout.NumpyArray(two),
                        awkward1.layout.NumpyArray(three),
                        awkward1.layout.NumpyArray(four),
                    ])

                    assert awkward1.to_list(ak_combined) == combined.tolist()
                    assert awkward1.to_numpy(ak_combined).dtype == combined.dtype

                    ak_combined = awkward1.layout.NumpyArray(one).mergemany([
                        awkward1.layout.NumpyArray(two),
                        awkward1.layout.EmptyArray(),
                        awkward1.layout.NumpyArray(four),
                    ])

                    assert awkward1.to_list(ak_combined) == combined.tolist()
                    assert awkward1.to_numpy(ak_combined).dtype == numpy.concatenate([one, two, four]).dtype

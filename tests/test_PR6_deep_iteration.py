# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy
# numba = pytest.importorskip("numba")

import awkward1

# py27 = 2 if sys.version_info[0] < 3 else 1

def test_iterator():
    content = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
    offsets = awkward1.layout.Index(numpy.array([0, 2, 2, 3], "i4"))
    array = awkward1.layout.ListOffsetArray(offsets, content)

    # for x in content:
    #     print(x)
    #
    # raise Exception

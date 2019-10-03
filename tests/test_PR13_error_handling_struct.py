# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy
numba = pytest.importorskip("numba")

import awkward1
awkward1_numba_util = pytest.importorskip("awkward1._numba.util")

py27 = (sys.version_info[0] < 3)

# def test_errors():
#     array.setid()
#     # with pytest.raises(ValueError):
#     #     array.content[20]
#
#     print(array.content[20])
#
#
#
#     raise Exception

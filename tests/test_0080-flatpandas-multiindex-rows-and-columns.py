# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

pandas = pytest.importorskip("pandas")

def test():
    array = awkward1.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])

    # print(awkward1.pandas.multiindex(array))
    # raise Exception

# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1

# def test():
#     array = awkward1.Array([
#         [3.3, 2.2, 5.5, 1.1, 4.4],
#         [4.4, 2.2, 1.1, 3.3, 5.5],
#         [2.2, 1.1, 4.4, 3.3, 5.5]]).layout
#     print(awkward1.tolist(array.argmin(axis=0)))
#     print(awkward1.tolist(array.argmin(axis=1)))
#     raise Exception
#
# #          0    1    2    3    4
# #          5    6    7    8    9
# #         10   11   12   13   14

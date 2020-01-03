# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import pytest
import numpy

import awkward1

# def test_parameters_on_arrays():
#     a = awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3, 4.4, 5.5]))
#     print(a.parameters)
#     print(a)
#     a.setparameter("one", ["two", 3, {"four": 5}])
#     print(a.parameters)
#     print(a)
#
#     b = awkward1.layout.ListOffsetArray64(awkward1.layout.Index64(numpy.array([0, 3, 3, 5], dtype=numpy.int64)), a)
#     print(b.parameters)
#     print(b)
#     b.setparameter("what", "ever")
#     print(b.parameters)
#     print(b)
#
#     raise Exception

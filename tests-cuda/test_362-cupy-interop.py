# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import itertools

import numpy

import awkward1
import cupy

def test_cupy_interop():
    c = cupy.arange(10)
    n = numpy.arange(10)
    cupy_index_arr = awkward1.layout.Index64.from_cupy(c)
    numpy_index_arr = awkward1.layout.Index64(n)

    # GPU->CPU
    assert awkward1.to_list(numpy.asarray(cupy_index_arr.copy_to("cpu"))) == awkward1.to_list(numpy.asarray(numpy_index_arr))
    # CPU->CPU
    assert awkward1.to_list(numpy.asarray(numpy_index_arr.copy_to("cpu"))) == awkward1.to_list(numpy.asarray(numpy_index_arr))
    #CPU->GPU->CPU
    assert awkward1.to_list(numpy.asarray(numpy_index_arr)) == awkward1.to_list(numpy.asarray(numpy_index_arr.copy_to("cuda").copy_to("cpu")))
    
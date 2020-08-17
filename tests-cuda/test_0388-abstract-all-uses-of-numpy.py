# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import cupy
import numpy
import awkward1


############################ array creation

def test_array():
    # data[, dtype=[, copy=]]
    pass

def test_asarray():
    # array[, dtype=]
    pass

def test_frombuffer():
    # array[, dtype=]
    pass

def test_zeros():
    # shape/len[, dtype=]
    pass

def test_ones():
    # shape/len[, dtype=]
    pass

def test_empty():
    # shape/len[, dtype=]
    pass

def test_full():
    # shape/len, value[, dtype=]
    pass

def test_arange():
    # stop[, dtype=]
    # start, stop[, dtype=]
    # start, stop, step[, dtype=]
    pass

def test_meshgrid():
    # *arrays, indexing="ij"
    pass

############################ testing

def test_array_equal():
    # array1, array2
    pass

def test_size():
    # array
    pass

def test_searchsorted():
    # stops, where, side="right"
    pass

############################ manipulation

def test_cumsum():
    # arrays[, out=]
    pass

def test_nonzero():
    # array
    pass

def test_unique():
    # array
    pass

def test_concatenate():
    # arrays
    pass

def test_repeat():
    # array, int
    # array1, array2
    pass

def test_stack():
    # arrays
    pass

def test_vstack():
    # arrays
    pass

def test_packbits():
    # array
    pass

def test_unpackbits():
    # array
    pass

def test_atleast_1d():
    # *arrays
    pass

def test_broadcast_to():
    # array, shape
    pass

############################ ufuncs

def test_sqrt():
    # array
    pass

def test_exp():
    # array
    pass

def test_true_divide():
    # array1, array2
    pass

def test_bitwise_or():
    # array1, array2[, out=output]
    pass

def test_logical_and():
    # array1, array2
    pass

def test_equal():
    # array1, array2
    pass

def test_ceil():
    # array
    pass

############################ reducers

def test_all():
    # array
    pass

def test_any():
    # array
    pass

def test_count_nonzero():
    # array
    pass

def test_sum():
    # array
    pass

def test_prod():
    # array
    pass

def test_min():
    # array
    pass

def test_max():
    # array
    pass

def test_argmin():
    # array[, axis=]
    pass

def test_argmax():
    # array[, axis=]
    pass

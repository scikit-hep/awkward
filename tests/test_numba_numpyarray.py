# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys

import numpy
import numba

import awkward1

def test():
    a = awkward1.layout.NumpyArray(numpy.arange(10))

    @numba.njit
    def stuff(q):
        return q

    stuff(a)

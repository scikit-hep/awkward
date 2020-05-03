# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest
import numpy

import awkward1
import awkward1._ext

def test():
    # rec = awkward1.zip({"x": awkward1.virtual(lambda: awkward1.Array([1, 2, 3, 4]), length=4),},
    #                    depth_limit=1)
    # q = rec.x[1:]
    # q.layout.generator()
    # q.layout.generator()

    # q = rec.layout["x"]
    # q = rec.x[1:].layout

    gen = awkward1._ext.ArrayGenerator(lambda: awkward1.layout.NumpyArray(numpy.array([1, 2, 3, 4])))
    # sgen = awkward1._ext.SliceGenerator(gen, slice(1, None))

    # g = sgen.generator
    # h = sgen.generator
    # print(g())
    # print(h())

    # print(sgen.generator)
    # print(sgen.generator)

    # raise Exception

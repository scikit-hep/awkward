# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import gc

import pytest

import numpy
import awkward1

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping


class Cache(MutableMapping):
    def __init__(self):
        self.data = {}
    def __getitem__(self, where):
        return self.data[where]
    def __setitem__(self, where, what):
        self.data[where] = what
    def __delitem__(self, where):
        del self.data[where]
    def __iter__(self):
        for x in self.data:
            yield x
    def __len__(self):
        return len(self.data)


def make_arrays():
    cache = Cache()
    x = awkward1.layout.VirtualArray(
        awkward1.layout.ArrayGenerator(
            lambda: awkward1.layout.NumpyArray(numpy.array([1.1, 2.2, 3.3]))
        ),
        awkward1.layout.ArrayCache(cache),
    )
    y = awkward1.layout.VirtualArray(
        awkward1.layout.ArrayGenerator(
            lambda: awkward1.layout.NumpyArray(numpy.array([100, 200, 300]))
        ),
        awkward1.layout.ArrayCache(cache),
    )
    inner = awkward1.layout.RecordArray({"x": x, "y": y})
    part = awkward1.partition.IrregularlyPartitionedArray(
        [inner], [3]
    )
    return awkward1.Array(part)


def test():
    z = make_arrays()
    repr(z)

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


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
    x = ak.layout.VirtualArray(
        ak.layout.ArrayGenerator(
            lambda: ak.layout.NumpyArray(np.array([1.1, 2.2, 3.3]))
        ),
        ak.layout.ArrayCache(cache),
    )
    y = ak.layout.VirtualArray(
        ak.layout.ArrayGenerator(
            lambda: ak.layout.NumpyArray(np.array([100, 200, 300]))
        ),
        ak.layout.ArrayCache(cache),
    )
    inner = ak.layout.RecordArray({"x": x, "y": y})
    part = ak.partition.IrregularlyPartitionedArray([inner], [3])
    return ak.Array(part)


def test():
    z = make_arrays()
    repr(z)

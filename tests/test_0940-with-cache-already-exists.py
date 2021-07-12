# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


def func():
    return ak.Array([1, 2, 3, 4, 5])


generator = ak.layout.ArrayGenerator(func, (), length=5)


def test():
    hold_cache = ak._util.MappingProxy({})
    cache = ak.layout.ArrayCache(hold_cache)
    layout = ak.layout.VirtualArray(generator, cache=cache)
    array = ak.Array(layout)

    cache_2 = {}
    array_2 = ak.with_cache(array, cache_2)
    ak.materialized(array_2)

    assert len(cache_2) > 0
    assert len(cache) == 0

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import pytest

import awkward1 as ak

def test_lazy_arrayset():
    array = ak.Array([
        {
            "listcollection": [
                {"item1": 1, "item2": 2},
                {"item1": 2, "item2": 4},
                {"item1": 3, "item2": 6},
            ],
            "collection": {"item1": 3, "item2": 4},
            "singleton": 5,
            "listsingleton": [1, 2, 3],
            "unioncollection": {"item1": 3},
            "masked": None,
        },
        {
            "listcollection": [
                {"item1": 1, "item2": 2},
                {"item1": 2, "item2": 4},
                {"item1": 3, "item2": 6},
            ],
            "collection": {"item1": 3, "item2": 4},
            "singleton": 5,
            "listsingleton": [1, 2, 3],
            "unioncollection": [{"item1": 2}],
            "masked": 4,
        },
        {
            "listcollection": [
                {"item1": 1, "item2": 2},
                {"item1": 2, "item2": 4},
                {"item1": 3, "item2": 6},
            ],
            "collection": {"item1": 3, "item2": 4},
            "singleton": 5,
            "listsingleton": [1, 2, 3],
            "unioncollection": {"item1": 4},
            "masked": 4,
        },
    ])

    class Canary(dict):
        def __init__(self):
            super().__init__()
            self.ops = []

        def __getitem__(self, key):
            self.ops.append(("get", key))
            return super().__getitem__(key)

        def __setitem__(self, key, value):
            self.ops.append(("set", key))
            return super().__setitem__(key, value)


    canary = Canary()
    prefix = "kitty"
    form, container, npart = ak.to_arrayset(array, container=canary, prefix=prefix)
    assert not any(op[0] == "get" for op in canary.ops)
    canary.ops.clear()

    cache = {}
    out = ak.from_arrayset(form, container, lazy=True, lazy_cache=cache, lazy_lengths=3, prefix=prefix, lazy_cache_key="hello")
    assert len(canary.ops) == 0
    assert len(cache) == 0

    assert len(out) == 3
    assert len(canary.ops) == 0
    assert len(cache) == 0

    assert ak.to_list(ak.num(out.listcollection)) == [3, 3, 3]
    assert canary.ops == [('get', 'kitty-node1-offsets')]
    assert set(cache) == {'hello', 'hello-kitty-node1-virtual'}
    canary.ops.clear()
    cache.clear()

    assert ak.to_list(out.unioncollection) == [{'item1': 3}, [{'item1': 2}], {'item1': 4}]
    assert canary.ops == [('get', 'kitty-node11-tags'), ('get', 'kitty-node11-index'), ('get', 'kitty-node14-offsets'), ('get', 'kitty-node13'), ('get', 'kitty-node16')]
    assert set(cache) == {'hello', 'hello-kitty-node11-virtual', 'hello-kitty-node13-virtual', 'hello-kitty-node16-virtual'}
    canary.ops.clear()
    cache.clear()

    assert ak.to_list(out.masked) == [None, 4, 4]
    assert canary.ops == [('get', 'kitty-node17-index'), ('get', 'kitty-node18')]
    assert set(cache) == {'hello', 'hello-kitty-node17-virtual'}
    canary.ops.clear()
    cache.clear()

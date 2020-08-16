# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys

import pytest

import numpy as np
import awkward1 as ak


class Canary(dict):
    def __init__(self):
        super(Canary, self).__init__()
        self.ops = []

    def __getitem__(self, key):
        self.ops.append(("get", key))
        return super(Canary, self).__getitem__(key)

    def __setitem__(self, key, value):
        self.ops.append(("set", key))
        return super(Canary, self).__setitem__(key, value)


def test_lazy_arrayset():
    array = ak.from_json("""
    [
        {
            "listcollection": [
                {"item1": 1, "item2": 2},
                {"item1": 2, "item2": 4},
                {"item1": 3, "item2": 6}
            ],
            "collection": {"item1": 3, "item2": 4},
            "singleton": 5,
            "listsingleton": [1, 2, 3],
            "unioncollection": {"item1": 3},
            "masked": null
        },
        {
            "listcollection": [
                {"item1": 1, "item2": 2},
                {"item1": 2, "item2": 4},
                {"item1": 3, "item2": 6}
            ],
            "collection": {"item1": 3, "item2": 4},
            "singleton": 5,
            "listsingleton": [1, 2, 3],
            "unioncollection": [{"item1": 2}],
            "masked": 4
        },
        {
            "listcollection": [
                {"item1": 1, "item2": 2},
                {"item1": 2, "item2": 4},
                {"item1": 3, "item2": 6}
            ],
            "collection": {"item1": 3, "item2": 4},
            "singleton": 5,
            "listsingleton": [1, 2, 3],
            "unioncollection": {"item1": 4},
            "masked": 4
        }
    ]""")

    canary = Canary()
    prefix = "kitty"
    form, container, npart = ak.to_arrayset(array, container=canary, prefix=prefix)
    assert not any(op[0] == "get" for op in canary.ops)
    canary.ops = []

    cache = {}
    out = ak.from_arrayset(form, container, lazy=True, lazy_cache=cache, lazy_lengths=3, prefix=prefix, lazy_cache_key="hello")
    assert len(canary.ops) == 0
    assert len(cache) == 0

    assert len(out) == 3
    assert len(canary.ops) == 0
    assert len(cache) == 0

    assert ak.to_list(ak.num(out.listcollection)) == [3, 3, 3]
    assert set(canary.ops) == {('get', 'kitty-node1-offsets')}
    assert set(cache) == {'hello', 'hello-kitty-node1-virtual'}
    canary.ops = []
    cache.clear()

    assert ak.to_list(out.unioncollection) == [{'item1': 3}, [{'item1': 2}], {'item1': 4}]
    assert set(canary.ops) == {('get', 'kitty-node11-tags'), ('get', 'kitty-node11-index'), ('get', 'kitty-node14-offsets'), ('get', 'kitty-node13'), ('get', 'kitty-node16')}
    assert set(cache) == {'hello', 'hello-kitty-node11-virtual', 'hello-kitty-node13-virtual', 'hello-kitty-node16-virtual'}
    canary.ops = []
    cache.clear()

    assert ak.to_list(out.masked) == [None, 4, 4]
    assert set(canary.ops) == {('get', 'kitty-node17-index'), ('get', 'kitty-node18')}
    assert set(cache) == {'hello', 'hello-kitty-node17-virtual'}
    canary.ops = []
    cache.clear()


def test_longer_than_expected():
    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64([0, 2, 4]),
            ak.layout.RecordArray({
                "item1": ak.layout.NumpyArray(np.arange(4)),
                "longitem": ak.layout.NumpyArray(np.arange(6)),
            }),
        )
    )
    out = ak.from_arrayset(*ak.to_arrayset(array), lazy=True, lazy_lengths=2)
    assert ak.to_list(out) == [[{'item1': 0, 'longitem': 0}, {'item1': 1, 'longitem': 1}], [{'item1': 2, 'longitem': 2}, {'item1': 3, 'longitem': 3}]]

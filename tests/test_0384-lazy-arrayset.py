# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

from __future__ import absolute_import

import pytest  # noqa: F401
import numpy as np  # noqa: F401
import awkward as ak  # noqa: F401


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


def test_array_builder():
    b = ak.ArrayBuilder()
    b.begin_list()

    b.begin_record()
    b.field("listcollection")

    b.begin_list()

    b.begin_record()
    b.field("item1")

    b.begin_list()
    b.integer(1)
    b.integer(3)
    b.integer(2)
    b.end_list()

    b.end_record()

    b.begin_record()
    b.field("item2")

    b.begin_list()
    b.integer(2)
    b.integer(6)
    b.integer(4)
    b.end_list()

    b.end_record()

    b.end_list()
    b.end_record()

    b.begin_record()
    b.field("collection")
    b.begin_list()
    b.begin_record()
    b.field("item1")
    b.integer(3)
    b.end_record()
    b.begin_record()
    b.field("item2")
    b.integer(4)
    b.end_record()

    b.end_list()
    b.end_record()

    b.begin_record()
    b.field("singleton")
    b.integer(5)
    b.end_record()

    b.begin_record()
    b.field("listsingleton")
    b.begin_list()
    b.integer(1)
    b.integer(3)
    b.integer(2)
    b.end_list()
    b.end_record()

    b.begin_record()
    b.field("unioncollection")
    b.begin_record()
    b.field("item1")
    b.integer(3)
    b.end_record()
    b.end_record()

    b.begin_record()
    b.field("masked")
    b.null()
    b.end_record()
    b.end_list()

    print(b.snapshot().layout)

    #
    #     },
    #     {
    #         "a_listcollection": [
    #             {"item1": 1, "item2": 2},
    #             {"item1": 2, "item2": 4},
    #             {"item1": 3, "item2": 6}
    #         ],
    #         "b_collection": {"item1": 3, "item2": 4},
    #         "c_singleton": 5,
    #         "d_listsingleton": [1, 2, 3],
    #         "e_unioncollection": [{"item1": 2}],
    #         "f_masked": 4
    #     },
    #     {
    #         "a_listcollection": [
    #             {"item1": 1, "item2": 2},
    #             {"item1": 2, "item2": 4},
    #             {"item1": 3, "item2": 6}
    #         ],
    #         "b_collection": {"item1": 3, "item2": 4},
    #         "c_singleton": 5,
    #         "d_listsingleton": [1, 2, 3],
    #         "e_unioncollection": {"item1": 4},
    #         "f_masked": 4
    #     }
    # ]


def test_lazy_buffers():

    array = ak.from_json(
        """
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
    ]"""
    )

    canary = Canary()
    key_format = "kitty-{form_key}-{attribute}"
    form, length, container = ak.to_buffers(
        array, container=canary, key_format=key_format
    )
    assert not any(op[0] == "get" for op in canary.ops)
    canary.ops = []

    cache = {}
    out = ak.from_buffers(
        form,
        length,
        container,
        key_format=key_format,
        lazy=True,
        lazy_cache=cache,
        lazy_cache_key="hello",
    )
    assert len(canary.ops) == 0
    assert len(cache) == 0

    assert len(out) == 3
    assert len(canary.ops) == 0
    assert len(cache) == 0

    assert ak.to_list(ak.num(out.listcollection)) == [3, 3, 3]
    assert set(canary.ops) == {("get", "kitty-node1-offsets")}
    assert "hello" in cache
    assert any(x.startswith("hello(kitty-node1-virtual") for x in cache)
    canary.ops = []
    cache.clear()

    assert ak.to_list(out.unioncollection) == [
        {"item1": 3},
        [{"item1": 2}],
        {"item1": 4},
    ]
    assert set(canary.ops) == {
        ("get", "kitty-node11-tags"),
        ("get", "kitty-node11-index"),
        ("get", "kitty-node14-offsets"),
        ("get", "kitty-node13-data"),
        ("get", "kitty-node16-data"),
    }
    assert "hello" in cache
    assert any(x.startswith("hello(kitty-node11-virtual") for x in cache)
    assert any(x.startswith("hello(kitty-node13-virtual") for x in cache)
    assert any(x.startswith("hello(kitty-node16-virtual") for x in cache)
    canary.ops = []
    cache.clear()

    assert ak.to_list(out.masked) == [None, 4, 4]
    assert set(canary.ops) == {
        ("get", "kitty-node17-index"),
        ("get", "kitty-node18-data"),
    }
    assert "hello" in cache
    assert any(x.startswith("hello(kitty-node17-virtual") for x in cache)
    canary.ops = []
    cache.clear()


def test_longer_than_expected():
    array = ak.Array(
        ak.layout.ListOffsetArray64(
            ak.layout.Index64([0, 2, 4]),
            ak.layout.RecordArray(
                {
                    "item1": ak.layout.NumpyArray(np.arange(4)),
                    "longitem": ak.layout.NumpyArray(np.arange(6)),
                }
            ),
        )
    )
    out = ak.from_buffers(*ak.to_buffers(array), lazy=True)
    assert ak.to_list(out) == [
        [{"item1": 0, "longitem": 0}, {"item1": 1, "longitem": 1}],
        [{"item1": 2, "longitem": 2}, {"item1": 3, "longitem": 3}],
    ]

# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

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
        print("get", where)
        return self.data[where]

    def __setitem__(self, where, what):
        print("put", where)
        self.data[where] = what

    def __delitem__(self, where):
        del self.data[where]

    def __iter__(self):
        for x in self.data:
            yield x

    def __len__(self):
        return len(self.data)


def test():
    array = ak.Array([[{"a": 1, "b" : [1, 2, 3]}], [{"a": 1, "b" : [4, 5]}, {"a" : 4, "b" : [2]}]])
    array_new = ak.Array(
        ak.layout.ListOffsetArray64(
            array.layout.offsets,
            ak.layout.RecordArray(
                {
                    "a": array.layout.content["a"],
                    "b": ak.layout.ListArray64(
                        array.layout.content["b"].offsets[:-1],
                        array.layout.content["b"].offsets[1:],
                        array.layout.content["b"].content
                    ),
                }
            )
        )
    )
    form, length, container = ak.to_buffers(array_new)
    reconstituted = ak.from_buffers(form, length, container, lazy=True, lazy_cache=Cache())
    assert reconstituted.tolist() == array_new.tolist()

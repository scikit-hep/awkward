# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpy_like import NumpyMetadata

np = NumpyMetadata.instance()
numpy = Numpy.instance()


class HashableDict:
    def __init__(self, obj):
        self.keys = tuple(sorted(obj))
        self.values = tuple(as_hashable(obj[k]) for k in self.keys)
        self.hash = hash((HashableDict, *self.keys), self.values)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return (
            isinstance(other, HashableDict)
            and self.keys == other.keys
            and self.values == other.values
        )


class HashableList:
    def __init__(self, obj):
        self.values = tuple(obj)
        self.hash = hash((HashableList, *self.values))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return isinstance(other, HashableList) and self.values == other.values


def as_hashable(obj):
    if isinstance(obj, dict):
        return HashableDict(obj)
    elif isinstance(obj, tuple):
        return tuple(as_hashable(x) for x in obj)
    elif isinstance(obj, list):
        return HashableList(obj)
    else:
        return obj

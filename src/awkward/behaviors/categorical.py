# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE
import awkward as ak
from awkward._behavior import behavior_of
from awkward._layout import wrap_layout
from awkward._nplikes import ufuncs
from awkward._nplikes.numpy import Numpy
from awkward._nplikes.numpylike import NumpyMetadata
from awkward.highlevel import Array

np = NumpyMetadata.instance()
numpy = Numpy.instance()


class CategoricalBehavior(Array):
    __name__ = "Array"


class _HashableDict:
    def __init__(self, obj):
        self.keys = tuple(sorted(obj))
        self.values = tuple(_as_hashable(obj[k]) for k in self.keys)
        self.hash = hash((_HashableDict, *self.keys), self.values)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return (
            isinstance(other, _HashableDict)
            and self.keys == other.keys
            and self.values == other.values
        )


class _HashableList:
    def __init__(self, obj):
        self.values = tuple(obj)
        self.hash = hash((_HashableList, *self.values))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return isinstance(other, _HashableList) and self.values == other.values


def _as_hashable(obj):
    if isinstance(obj, dict):
        return _HashableDict(obj)
    elif isinstance(obj, tuple):
        return tuple(_as_hashable(x) for x in obj)
    elif isinstance(obj, list):
        return _HashableList(obj)
    else:
        return obj


def _categorical_equal(one, two):
    behavior = behavior_of(one, two)

    one, two = one.layout, two.layout

    assert one.is_indexed or (one.is_option and one.is_indexed)
    assert two.is_indexed or (two.is_option and two.is_indexed)
    assert one.parameter("__array__") == "categorical"
    assert two.parameter("__array__") == "categorical"

    one_index = numpy.asarray(one.index)
    two_index = numpy.asarray(two.index)
    one_content = wrap_layout(one.content, behavior)
    two_content = wrap_layout(two.content, behavior)

    if len(one_content) == len(two_content) and ak.operations.all(
        one_content == two_content, axis=None
    ):
        one_mapped = one_index

    else:
        one_list = ak.operations.to_list(one_content)
        two_list = ak.operations.to_list(two_content)
        one_hashable = [_as_hashable(x) for x in one_list]
        two_hashable = [_as_hashable(x) for x in two_list]
        two_lookup = {x: i for i, x in enumerate(two_hashable)}

        one_to_two = numpy.empty(len(one_hashable) + 1, dtype=np.int64)
        for i, x in enumerate(one_hashable):
            one_to_two[i] = two_lookup.get(x, len(two_hashable))
        one_to_two[-1] = -1

        one_mapped = one_to_two[one_index]

    out = one_mapped == two_index
    out = wrap_layout(ak.contents.NumpyArray(out), behavior_of(one, two))
    return out


def _apply_ufunc(ufunc, method, inputs, kwargs):
    nextinputs = []
    for x in inputs:
        if isinstance(x, ak.highlevel.Array) and x.layout.is_indexed:
            nextinputs.append(
                ak.highlevel.Array(x.layout.project(), behavior=behavior_of(x))
            )
        else:
            nextinputs.append(x)

    return getattr(ufunc, method)(*nextinputs, **kwargs)


def register(behavior):
    behavior["categorical"] = CategoricalBehavior
    behavior[ufuncs.equal, "categorical", "categorical"] = _categorical_equal
    behavior[ufuncs.ufunc, "categorical"] = _apply_ufunc

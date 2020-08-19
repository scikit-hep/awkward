# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import awkward1.highlevel
import awkward1.nplike
import awkward1.operations.convert
import awkward1.operations.reducers
import awkward1._util


np = awkward1.nplike.NumpyMetadata.instance()


class CategoricalBehavior(awkward1.highlevel.Array):
    __name__ = "Array"


awkward1.behavior["categorical"] = CategoricalBehavior


class _HashableDict(object):
    def __init__(self, obj):
        self.keys = tuple(sorted(obj))
        self.values = tuple(_hashable(obj[k]) for k in self.keys)
        self.hash = hash((_HashableDict,) + self.keys, self.values)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return (
            isinstance(other, _HashableDict)
            and self.keys == other.keys
            and self.values == other.values
        )


class _HashableList(object):
    def __init__(self, obj):
        self.values = tuple(obj)
        self.hash = hash((_HashableList,) + self.values)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return (
            isinstance(other, _HashableList)
            and self.values == other.values
        )


def _hashable(obj):
    if isinstance(obj, dict):
        return _HashableDict(obj)
    elif isinstance(obj, tuple):
        return tuple(_hashable(x) for x in obj)
    elif isinstance(obj, list):
        return _HashableList(obj)
    else:
        return obj


def _categorical_equal(one, two):
    behavior = awkward1._util.behaviorof(one, two)

    one, two = one.layout, two.layout

    assert isinstance(
        one, awkward1._util.indexedtypes + awkward1._util.indexedoptiontypes
    )
    assert isinstance(
        two, awkward1._util.indexedtypes + awkward1._util.indexedoptiontypes
    )
    assert one.parameter("__array__") == "categorical"
    assert two.parameter("__array__") == "categorical"

    one_index = awkward1.nplike.numpy.asarray(one.index)
    two_index = awkward1.nplike.numpy.asarray(two.index)
    one_content = awkward1._util.wrap(one.content, behavior)
    two_content = awkward1._util.wrap(two.content, behavior)

    if (
        len(one_content) == len(two_content) and
        awkward1.operations.reducers.all(one_content == two_content, axis=None)
    ):
        one_mapped = one_index

    else:
        one_list = awkward1.operations.convert.to_list(one_content)
        two_list = awkward1.operations.convert.to_list(two_content)
        one_hashable = [_hashable(x) for x in one_list]
        two_hashable = [_hashable(x) for x in two_list]
        two_lookup = {x: i for i, x in enumerate(two_hashable)}

        one_to_two = awkward1.nplike.numpy.empty(len(one_hashable) + 1, dtype=np.int64)
        for i, x in enumerate(one_hashable):
            one_to_two[i] = two_lookup.get(x, len(two_hashable))
        one_to_two[-1] = -1

        one_mapped = one_to_two[one_index]

    out = one_mapped == two_index
    return awkward1._util.wrap(
        awkward1.layout.NumpyArray(out), awkward1._util.behaviorof(one, two)
    )


awkward1.behavior[awkward1.nplike.numpy.equal, "categorical", "categorical"] = _categorical_equal


def to_categorical(array, highlevel=True):
    def getfunction(layout, depth):
        p = layout.purelist_parameter("__array__")
        print("depth", depth, "layout", type(layout), "p", p)

        if (layout.purelist_depth == 1 or (
            layout.purelist_depth == 2 and (p == "string" or p == "bytestring")
        )):
            if isinstance(layout, awkward1._util.optiontypes):
                layout = layout.simplify()
            if isinstance(
                layout,
                awkward1._util.indexedoptiontypes + awkward1._util.indexedtypes
            ):
                content = layout.content
                cls = awkward1.layout.IndexedOptionArray64
            elif isinstance(layout, awkward1._util.optiontypes):
                content = layout.content
                cls = awkward1.layout.IndexedOptionArray64
            else:
                content = layout
                cls = awkward1.layout.IndexedArray64

            content_list = awkward1.operations.convert.to_list(content)
            hashable = [_hashable(x) for x in content_list]

            lookup = {}
            is_first = awkward1.nplike.numpy.empty(len(hashable), dtype=np.bool_)
            mapping = awkward1.nplike.numpy.empty(len(hashable), dtype=np.int64)
            for i, x in enumerate(hashable):
                if x in lookup:
                    is_first[i] = False
                    mapping[i] = lookup[x]
                else:
                    is_first[i] = True
                    lookup[x] = i
                    mapping[i] = i

            if isinstance(layout, awkward1._util.indexedoptiontypes):
                original_index = awkward1.nplike.numpy.asarray(layout.index)
                index = mapping[original_index]
                index[original_index < 0] = -1
                index = awkward1.layout.Index64(index)

            elif isinstance(layout, awkward1._util.indexedtypes):
                original_index = awkward1.nplike.numpy.asarray(layout.index)
                index = awkward1.layout.Index64(mapping[original_index])

            elif isinstance(layout, awkward1._util.optiontypes):
                mask = awkward1.nplike.numpy.asarray(layout.bytemask())
                mapping[mask.view(np.bool_)] = -1
                index = awkward1.layout.Index64(mapping)

            else:
                index = awkward1.layout.Index64(mapping)

            out = cls(index, content[is_first], parameters={"__array__": "categorical"})
            return lambda: out

        else:
            return None

    out = awkward1._util.recursively_apply(
        awkward1.operations.convert.to_layout(
            array, allow_record=False, allow_other=False
        ),
        getfunction,
    )
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out


__all__ = [
    x
    for x in list(globals())
    if not x.startswith("_")
    and x not in (
        "absolute_import",
        "np",
        "awkward1",
    )
]

# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import numbers
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1._util
import awkward1.layout
import awkward1._numpy
import awkward1.operations.convert

def withfield(base, what, where=None):
    base = awkward1.operations.convert.tolayout(base, allowrecord=True, allowother=False)
    what = awkward1.operations.convert.tolayout(what, allowrecord=True, allowother=True)

    def getfunction(inputs):
        base, what = inputs
        if isinstance(base, awkward1.layout.RecordArray):
            if not isinstance(what, awkward1.layout.Content):
                what = awkward1.layout.NumpyArray(numpy.lib.stride_tricks.as_strided([what], shape=(len(base),), strides=(0,)))
            return lambda depth: base.setitem_field(where, what)
        else:
            return None

    out = awkward1._util.broadcast_and_apply([base, what], getfunction)
    return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(base, what))

def isna(array):
    import awkward1.highlevel

    def apply(layout):
        if isinstance(layout, awkward1._util.unknowntypes):
            return apply(awkward1.layout.NumpyArray(numpy.array([])))

        elif isinstance(layout, awkward1._util.indexedtypes):
            return apply(layout.project())

        elif isinstance(layout, awkward1._util.uniontypes):
            contents = [apply(layout.project(i)) for i in range(layout.numcontents)]
            out = numpy.empty(len(layout), dtype=numpy.bool_)
            tags = numpy.asarray(layout.tags)
            for tag, content in enumerate(contents):
                out[tags == tag] = content
            return out

        elif isinstance(layout, awkward1._util.optiontypes):
            index = numpy.asarray(layout.index)
            return (index < 0)

        else:
            return numpy.zeros(len(layout), dtype=numpy.bool_)

    out = apply(awkward1.operations.convert.tolayout(array, allowrecord=False))
    return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))

def notna(array):
    return ~isna(array)

@awkward1._numpy.implements(numpy.size)
def size(array, axis=None):
    if axis is not None and axis < 0:
        raise NotImplementedError("ak.size with axis < 0")

    def recurse(layout, axis, sizes):
        if isinstance(layout, awkward1._util.unknowntypes):
            pass
        elif isinstance(layout, awkward1._util.indexedtypes):
            recurse(layout.content, axis, sizes)
        elif isinstance(layout, awkward1._util.uniontypes):
            compare = None
            for x in layout.contents:
                inner = []
                recurse(x, axis, inner)
                if compare is None:
                    compare = inner
                elif compare != inner:
                    raise ValueError("ak.size is ambiguous due to union of different sizes")
            sizes.extend(compare)
        elif isinstance(layout, awkward1._util.optiontypes):
            return recurse(layout.content, axis, sizes)
        elif isinstance(layout, awkward1._util.listtypes):
            if isinstance(layout, awkward1.layout.RegularArray):
                sizes.append(layout.size)
            else:
                sizes.append(None)
            if axis is None:
                recurse(layout.content, axis, sizes)
            elif axis > 0:
                recurse(layout.content, axis - 1, sizes)
        elif isinstance(layout, awkward1._util.recordtypes):
            compare = None
            for x in layout.contents:
                inner = []
                recurse(x, axis, inner)
                if compare is None:
                    compare = inner
                elif compare != inner:
                    raise ValueError("ak.size is ambiguous due to record of different sizes")
            sizes.extend(compare)
        elif isinstance(layout, awkward1.layout.NumpyArray):
            if axis is None:
                sizes.extend(numpy.asarray(layout).shape[1:])
            else:
                sizes.extend(numpy.asarray(layout).shape[1:axis + 2])
        else:
            raise AssertionError("unrecognized Content type")

    layout = awkward1.operations.convert.tolayout(array, allowrecord=False)
    layout = awkward1.layout.RegularArray(layout, len(layout))

    sizes = []
    recurse(layout, axis, sizes)

    if axis is None:
        out = 1
        for size in sizes:
            if size is None:
                raise ValueError("ak.size is ambiguous due to variable-length arrays (try ak.flatten to remove structure or ak.tonumpy to force regularity, if possible)")
            else:
                out *= size
        return out
    else:
        if sizes[-1] is None:
            raise ValueError("ak.size is ambiguous due to variable-length arrays at axis {0} (try ak.flatten to remove structure or ak.tonumpy to force regularity, if possible)".format(axis))
        else:
            return sizes[-1]

@awkward1._numpy.implements(numpy.atleast_1d)
def atleast_1d(*arrays):
    return numpy.atleast_1d(*[awkward1.operations.convert.tonumpy(x) for x in arrays])

@awkward1._numpy.implements(numpy.concatenate)
def concatenate(arrays, axis=0, mergebool=True):
    import awkward1.highlevel

    if axis != 0:
        raise NotImplementedError("axis={0}".format(axis))

    contents = [awkward1.operations.convert.tolayout(x, allowrecord=False) for x in arrays]

    if len(contents) == 0:
        raise ValueError("need at least one array to concatenate")
    out = contents[0]
    for x in contents[1:]:
        if not out.mergeable(x, mergebool=mergebool):
            out = out.merge_as_union(x)
        else:
            out = out.merge(x)
        if isinstance(out, awkward1._util.uniontypes):
            out = out.simplify(mergebool=mergebool)

    return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(*arrays))

@awkward1._numpy.implements(numpy.where)
def where(condition, *args, **kwargs):
    import awkward1.highlevel

    condition = awkward1.operations.convert.tonumpy(condition)

    if len(args) == 0:
        out = numpy.nonzero(condition)
        return tuple(awkward1.highlevel.Array(x) for x in out)

    elif len(args) == 1:
        raise ValueError("either both or neither of x and y should be given")

    elif len(args) == 2:
        if len(condition.shape) != 1:
            raise NotImplementedError("FIXME: ak.where(condition, x, y) where condition is not 1-d")

        x = awkward1.operations.convert.tolayout(args[0], allowrecord=False)
        y = awkward1.operations.convert.tolayout(args[1], allowrecord=False)
        mergebool, = awkward1._util.extra((), kwargs, [
            ("mergebool", True)])

        tags = (condition == 0)
        assert tags.itemsize == 1
        index = numpy.empty(len(tags), dtype=numpy.int64)
        index = numpy.arange(len(condition), dtype=numpy.int64)

        tags = awkward1.layout.Index8(tags.view(numpy.int8))
        index = awkward1.layout.Index64(index)
        tmp = awkward1.layout.UnionArray8_64(tags, index, [x, y])
        out = tmp.simplify(mergebool=mergebool)

        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(*((condition,) + args)))

    else:
        raise TypeError("where() takes from 1 to 3 positional arguments but {0} were given".format(len(args) + 1))

def sizes(array, axis=0):
    behavior = awkward1._util.behaviorof(array)
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    return awkward1._util.wrap(layout.sizes(axis), behavior)

def flatten(array, axis=0):
    behavior = awkward1._util.behaviorof(array)
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    return awkward1._util.wrap(layout.flatten(axis), behavior)

def count(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])
        return reduce([numpy.size(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.count(axis=axis, mask=False, keepdims=keepdims), behavior)

def count_nonzero(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])
        return reduce([numpy.count_nonzero(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.count_nonzero(axis=axis, mask=False, keepdims=keepdims), behavior)

def sum(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] + reduce(xs[1:])
        return reduce([numpy.sum(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.sum(axis=axis, mask=False, keepdims=keepdims), behavior)

def prod(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] * reduce(xs[1:])
        return reduce([numpy.prod(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.prod(axis=axis, mask=False, keepdims=keepdims), behavior)

def any(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] or reduce(xs[1:])
        return reduce([numpy.any(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.any(axis=axis, mask=False, keepdims=keepdims), behavior)

def all(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                return xs[0] and reduce(xs[1:])
        return reduce([numpy.all(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.all(axis=axis, mask=False, keepdims=keepdims), behavior)

def min(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                x, y = xs[0], reduce(xs[1:])
                return x if x < y else y
        return reduce([numpy.min(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.min(axis=axis, mask=True, keepdims=keepdims), behavior)

def max(array, axis=None, keepdims=False):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if axis is None:
        def reduce(xs):
            if len(xs) == 1:
                return xs[0]
            else:
                x, y = xs[0], reduce(xs[1:])
                return x if x > y else y
        return reduce([numpy.max(x) for x in awkward1._util.completely_flatten(layout)])
    else:
        behavior = awkward1._util.behaviorof(array)
        return awkward1._util.wrap(layout.max(axis=axis, mask=True, keepdims=keepdims), behavior)

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numbers", "Iterable", "numpy", "awkward1")]

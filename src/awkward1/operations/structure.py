# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import awkward1._util
import awkward1.layout
import awkward1._connect._numpy
import awkward1.operations.convert

def withfield(base, what, where=None):
    base = awkward1.operations.convert.tolayout(base, allowrecord=True, allowother=False)
    what = awkward1.operations.convert.tolayout(what, allowrecord=True, allowother=True)

    def getfunction(inputs, depth):
        base, what = inputs
        if isinstance(base, awkward1.layout.RecordArray):
            if not isinstance(what, awkward1.layout.Content):
                what = awkward1.layout.NumpyArray(numpy.lib.stride_tricks.as_strided([what], shape=(len(base),), strides=(0,)))
            return lambda: (base.setitem_field(where, what),)
        else:
            return None

    out = awkward1._util.broadcast_and_apply([base, what], getfunction)
    assert isinstance(out, tuple) and len(out) == 1
    return awkward1._util.wrap(out[0], behavior=awkward1._util.behaviorof(base, what))

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

def num(array, axis=1):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False)
    out = layout.num(axis=axis)
    return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))

@awkward1._connect._numpy.implements(numpy.size)
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

@awkward1._connect._numpy.implements(numpy.atleast_1d)
def atleast_1d(*arrays):
    return numpy.atleast_1d(*[awkward1.operations.convert.tonumpy(x) for x in arrays])

@awkward1._connect._numpy.implements(numpy.concatenate)
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

@awkward1._connect._numpy.implements(numpy.broadcast_arrays)
def broadcast_arrays(*arrays):
    inputs = [awkward1.operations.convert.tolayout(x, allowrecord=True, allowother=False) for x in arrays]

    def getfunction(inputs, depth):
        if all(isinstance(x, awkward1.layout.NumpyArray) for x in inputs):
            return lambda: tuple(inputs)
        else:
            return None

    out = awkward1._util.broadcast_and_apply(inputs, getfunction)
    assert isinstance(out, tuple)
    return [awkward1._util.wrap(x, awkward1._util.behaviorof(arrays)) for x in out]

@awkward1._connect._numpy.implements(numpy.where)
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

def flatten(array, axis=1):
    behavior = awkward1._util.behaviorof(array)
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    return awkward1._util.wrap(layout.flatten(axis), behavior)

def rpad(array, length, axis=1, clip=False):
    behavior = awkward1._util.behaviorof(array)
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if clip:
        return awkward1._util.wrap(layout.rpad_and_clip(length, axis), behavior)
    else:
        return awkward1._util.wrap(layout.rpad(length, axis), behavior)

def zip(arrays, depthlimit=None, parameters=None):
    if depthlimit is not None and depthlimit <= 0:
        raise ValueError("depthlimit must be None or at least 1")

    behavior = awkward1._util.behaviorof(*arrays)
    if isinstance(arrays, dict):
        recordlookup = []
        layouts = []
        for n, x in arrays.items():
            recordlookup.append(n)
            layouts.append(awkward1.operations.convert.tolayout(x, allowrecord=False, allowother=False))
    else:
        recordlookup = None
        layouts = []
        for x in arrays:
            layouts.append(awkward1.operations.convert.tolayout(x, allowrecord=False, allowother=False))

    def getfunction(inputs, depth):
        if (depthlimit is None and any(x.purelist_depth == 1 for x in inputs)) or (depthlimit == depth):
            return lambda: (awkward1.layout.RecordArray(inputs, recordlookup, parameters=parameters),)
        else:
            return None

    out = awkward1._util.broadcast_and_apply(layouts, getfunction)
    assert isinstance(out, tuple) and len(out) == 1
    return awkward1._util.wrap(out[0], behavior)

def unzip(array):
    keys = awkward1.operations.describe.keys(array)
    if len(keys) == 0:
        return (array,)
    else:
        return tuple(array[n] for n in keys)

# def cross(arrays, axis=1, nested=None, parameters=None):
#     if axis < 0:
#         raise ValueError("cross 'axis' must be non-negative")
#
#     elif axis == 0:
#         raise NotImplementedError
#
#     else:



__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numpy", "awkward1")]

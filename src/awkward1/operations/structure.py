# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import awkward1._util
import awkward1.layout
import awkward1._connect._numpy
import awkward1.operations.convert

def withfield(base, what, where=None, highlevel=True):
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
    if highlevel:
        return awkward1._util.wrap(out[0], behavior=awkward1._util.behaviorof(base, what))
    else:
        return out[0]

def isna(array, highlevel=True):
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
    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
    else:
        return out

def notna(array, highlevel=True):
    return ~isna(array, highlevel=highlevel)

def num(array, axis=1, highlevel=True):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False)
    out = layout.num(axis=axis)
    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
    else:
        return out

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
def concatenate(arrays, axis=0, mergebool=True, highlevel=True):
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

    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(*arrays))
    else:
        return out

@awkward1._connect._numpy.implements(numpy.broadcast_arrays)
def broadcast_arrays(*arrays, **kwargs):
    highlevel, = awkward1._util.extra((), kwargs, [
        ("highlevel", True)])

    inputs = [awkward1.operations.convert.tolayout(x, allowrecord=True, allowother=False) for x in arrays]

    def getfunction(inputs, depth):
        if all(isinstance(x, awkward1.layout.NumpyArray) for x in inputs):
            return lambda: tuple(inputs)
        else:
            return None

    out = awkward1._util.broadcast_and_apply(inputs, getfunction)
    assert isinstance(out, tuple)
    if highlevel:
        return [awkward1._util.wrap(x, awkward1._util.behaviorof(arrays)) for x in out]
    else:
        return list(out)

@awkward1._connect._numpy.implements(numpy.where)
def where(condition, *args, **kwargs):
    mergebool, highlevel = awkward1._util.extra((), kwargs, [
        ("mergebool", True),
        ("highlevel", True)])

    npcondition = awkward1.operations.convert.tonumpy(condition)

    if len(args) == 0:
        out = numpy.nonzero(npcondition)
        if highlevel:
            return tuple(awkward1._util.wrap(awkward1.layout.NumpyArray(x), awkward1._util.behaviorof(condition)) for x in out)
        else:
            return tuple(awkward1.layout.NumpyArray(x) for x in out)

    elif len(args) == 1:
        raise ValueError("either both or neither of x and y should be given")

    elif len(args) == 2:
        if len(npcondition.shape) != 1:
            raise NotImplementedError("FIXME: ak.where(condition, x, y) where condition is not 1-d")

        x = awkward1.operations.convert.tolayout(args[0], allowrecord=False)
        y = awkward1.operations.convert.tolayout(args[1], allowrecord=False)

        tags = (npcondition == 0)
        assert tags.itemsize == 1
        index = numpy.empty(len(tags), dtype=numpy.int64)
        index = numpy.arange(len(npcondition), dtype=numpy.int64)

        tags = awkward1.layout.Index8(tags.view(numpy.int8))
        index = awkward1.layout.Index64(index)
        tmp = awkward1.layout.UnionArray8_64(tags, index, [x, y])
        out = tmp.simplify(mergebool=mergebool)

        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(*((npcondition,) + args)))

    else:
        raise TypeError("where() takes from 1 to 3 positional arguments but {0} were given".format(len(args) + 1))

def flatten(array, axis=1, highlevel=True):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    out = layout.flatten(axis)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out

def rpad(array, length, axis=1, clip=False, highlevel=True):
    layout = awkward1.operations.convert.tolayout(array, allowrecord=False, allowother=False)
    if clip:
        out = layout.rpad_and_clip(length, axis)
    else:
        out = layout.rpad(length, axis)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out

def zip(arrays, depthlimit=None, parameters=None, highlevel=True):
    if depthlimit is not None and depthlimit <= 0:
        raise ValueError("depthlimit must be None or at least 1")

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
    if highlevel:
        return awkward1._util.wrap(out[0], awkward1._util.behaviorof(*arrays))
    else:
        return out[0]

def unzip(array):
    keys = awkward1.operations.describe.keys(array)
    if len(keys) == 0:
        return (array,)
    else:
        return tuple(array[n] for n in keys)

def argcross(arrays, axis=1, nested=None, parameters=None, highlevel=True):
    if axis < 0:
        raise ValueError("argcross's 'axis' must be non-negative")

    else:
        if isinstance(arrays, dict):
            layouts = dict((n, awkward1.operations.convert.tolayout(x, allowrecord=False, allowother=False).localindex(axis)) for n, x in arrays.items())
        else:
            layouts = [awkward1.operations.convert.tolayout(x, allowrecord=False, allowother=False).localindex(axis) for x in arrays]

        result = cross(layouts, axis=axis, nested=nested, parameters=parameters, highlevel=False)

        if highlevel:
            return awkward1._util.wrap(result, awkward1._util.behaviorof(*arrays))
        else:
            return result

def cross(arrays, axis=1, nested=None, parameters=None, highlevel=True):
    if axis < 0:
        raise ValueError("cross's 'axis' must be non-negative")

    elif axis == 0:
        if nested is None or nested is False:
            nested = []

        if isinstance(arrays, dict):
            if nested is True:
                nested = list(arrays.keys())   # includes the last key, but it's ignored below
            if any(not (isinstance(n, str) and n in arrays) for x in nested):
                raise ValueError("cross's 'nested' must be dict keys for a dict of arrays")
            recordlookup = []
            layouts = []
            tonested = []
            for i, (n, x) in enumerate(arrays.items()):
                recordlookup.append(n)
                layouts.append(awkward1.operations.convert.tolayout(x, allowrecord=False, allowother=False))
                if n in nested:
                    tonested.append(i)
            nested = tonested

        else:
            if nested is True:
                nested = list(range(len(arrays) - 1))
            if any(not (isinstance(x, int) and 0 <= x < len(arrays) - 1) for x in nested):
                raise ValueError("cross's 'nested' must be integers in [0, len(arrays) - 1) for an iterable of arrays")
            recordlookup = None
            layouts = []
            for x in arrays:
                layouts.append(awkward1.operations.convert.tolayout(x, allowrecord=False, allowother=False))

        indexes = [awkward1.layout.Index64(x.reshape(-1)) for x in numpy.meshgrid(*[numpy.arange(len(x), dtype=numpy.int64) for x in layouts], indexing="ij")]
        outs = [awkward1.layout.IndexedArray64(x, y) for x, y in __builtins__["zip"](indexes, layouts)]

        result = awkward1.layout.RecordArray(outs, recordlookup, parameters=parameters)
        for i in range(len(arrays) - 1, -1, -1):
            if i in nested:
                result = awkward1.layout.RegularArray(result, len(layouts[i + 1]))

    else:
        def newaxis(layout, i):
            if i == 0:
                return layout
            else:
                return awkward1.layout.RegularArray(newaxis(layout, i - 1), 1)

        def getfunction1(layout, depth, i):
            if depth == 2:
                return lambda: newaxis(layout, i)
            else:
                return None

        def getfunction2(layout, depth, i):
            if depth == axis:
                inside = len(arrays) - i - 1
                outside = i
                return lambda: newaxis(awkward1._util.recursively_apply(layout, getfunction1, args=(inside,)), outside)
            else:
                return None

        def apply(x, i):
            return awkward1._util.recursively_apply(awkward1.operations.convert.tolayout(x, allowrecord=False, allowother=False), getfunction2, args=(i,))

        toflatten = []
        if nested is None or nested is False:
            nested = []

        if isinstance(arrays, dict):
            if nested is True:
                nested = list(arrays.keys())   # includes the last key, but it's ignored below
            if any(not (isinstance(n, str) and n in arrays) for x in nested):
                raise ValueError("cross's 'nested' must be dict keys for a dict of arrays")
            recordlookup = []
            layouts = []
            for i, (n, x) in enumerate(arrays.items()):
                recordlookup.append(n)
                layouts.append(apply(x, i))
                if i < len(arrays) - 1 and n not in nested:
                    toflatten.append(axis + i + 1)

        else:
            if nested is True:
                nested = list(range(len(arrays) - 1))
            if any(not (isinstance(x, int) and 0 <= x < len(arrays) - 1) for x in nested):
                raise ValueError("cross's 'nested' must be integers in [0, len(arrays) - 1) for an iterable of arrays")
            recordlookup = None
            layouts = []
            for i, x in enumerate(arrays):
                layouts.append(apply(x, i))
                if i < len(arrays) - 1 and i not in nested:
                    toflatten.append(axis + i + 1)

        def getfunction3(inputs, depth):
            if depth == axis + len(arrays):
                return lambda: (awkward1.layout.RecordArray(inputs, recordlookup, parameters=parameters),)
            else:
                return None

        out = awkward1._util.broadcast_and_apply(layouts, getfunction3)
        assert isinstance(out, tuple) and len(out) == 1
        result = out[0]

        while len(toflatten) != 0:
            axis = toflatten.pop()
            result = flatten(result, axis=axis, highlevel=False)

    if highlevel:
        return awkward1._util.wrap(result, awkward1._util.behaviorof(*arrays))
    else:
        return result

def choose(array, n, axis=1, nested=None, keys=None, parameters=None, highlevel=True):
    if nested is not None:
        raise NotImplementedError
    if parameters is None:
        parameters = {}

    layout = awkward1.operations.convert.tolayout(array, allowrecord=False)
    out = layout.choose(n, keys=keys, parameters=parameters, axis=axis)
    if highlevel:
        return awkward1._util.wrap(out, behavior=awkward1._util.behaviorof(array))
    else:
        return out

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numpy", "awkward1")]

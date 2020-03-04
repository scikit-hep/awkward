# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

from __future__ import absolute_import

import sys
import numbers
import json
import collections
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1.layout
import awkward1._io
import awkward1._util

def fromnumpy(array, regulararray=False, highlevel=True, behavior=None):
    def recurse(array, index):
        if regulararray and len(array.shape) > 1:
            return awkward1.layout.RegularArray(recurse(array.reshape((-1,) + array.shape[2:]), index), array.shape[1])

        if len(array.shape) == 0:
            data = awkward1.layout.NumpyArray(array.reshape(1))
        else:
            data = awkward1.layout.NumpyArray(array)

        if index is not None:
            return awkward1.layout.IndexedOptionArray64(index, data)
        else:
            return data

    if isinstance(array, numpy.ma.MaskedArray):
        mask = numpy.ma.getmaskarray(array)
        array = numpy.ma.getdata(array)
        index = numpy.arange(array.size, dtype=numpy.int64)
        index[mask.reshape(-1)] = -1
        index = awkward1.layout.Index64(index)
        if len(mask.shape) > 1:
            regulararray = True
    else:
        index = None

    layout = recurse(array, index)
    if highlevel:
        return awkward1._util.wrap(layout, behavior)
    else:
        return layout

def fromiter(iterable, highlevel=True, behavior=None, initial=1024, resize=2.0):
    out = awkward1.layout.ArrayBuilder(initial=initial, resize=resize)
    for x in iterable:
        out.fromiter(x)
    layout = out.snapshot()
    if highlevel:
        return awkward1._util.wrap(layout, behavior)
    else:
        return layout

def fromjson(source, highlevel=True, behavior=None, initial=1024, resize=2.0, buffersize=65536):
    layout = awkward1._io.fromjson(source, initial=initial, resize=resize, buffersize=buffersize)
    if highlevel:
        return awkward1._util.wrap(layout, behavior)
    else:
        return layout

def tonumpy(array):
    import awkward1.highlevel

    if isinstance(array, (bool, str, bytes, numbers.Number)):
        return numpy.array([array])[0]

    elif sys.version_info[0] < 3 and isinstance(array, unicode):
        return numpy.array([array])[0]

    elif isinstance(array, numpy.ndarray):
        return array

    elif isinstance(array, awkward1.highlevel.Array):
        return tonumpy(array.layout)

    elif isinstance(array, awkward1.highlevel.Record):
        out = array.layout
        return tonumpy(out.array[out.at : out.at + 1])[0]

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return tonumpy(array.snapshot().layout)

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return tonumpy(array.snapshot())

    elif awkward1.operations.describe.parameters(array).get("__array__") == "char":
        if awkward1.operations.describe.parameters(array).get("encoding") is None:
            return tonumpy(array.__bytes__())
        else:
            return tonumpy(array.__str__())

    elif awkward1.operations.describe.parameters(array).get("__array__") == "string":
        if awkward1.operations.describe.parameters(array.content).get("encoding") is None:
            return numpy.array([awkward1.behaviors.string.CharBehavior(array[i]).__bytes__() for i in range(len(array))])
        else:
            return numpy.array([awkward1.behaviors.string.CharBehavior(array[i]).__str__() for i in range(len(array))])

    elif isinstance(array, awkward1._util.unknowntypes):
        return numpy.array([])

    elif isinstance(array, awkward1._util.indexedtypes):
        return tonumpy(array.project())

    elif isinstance(array, awkward1._util.uniontypes):
        contents = [tonumpy(array.project(i)) for i in range(array.numcontents)]
        try:
            out = numpy.concatenate(contents)
        except:
            raise ValueError("cannot convert {0} into numpy.ndarray".format(array))
        tags = numpy.asarray(array.tags)
        for tag, content in enumerate(contents):
            mask = (tags == tag)
            out[mask] = content
        return out

    elif isinstance(array, awkward1._util.optiontypes):
        content = tonumpy(array.project())
        shape = list(content.shape)
        shape[0] = len(array)
        data = numpy.empty(shape, dtype=content.dtype)
        mask0 = (numpy.asarray(array.index) < 0)
        mask = numpy.broadcast_to(mask0.reshape((shape[0],) + (1,)*(len(shape) - 1)), shape)
        data[~mask0] = content
        return numpy.ma.MaskedArray(data, mask)

    elif isinstance(array, awkward1.layout.RegularArray):
        out = tonumpy(array.content)
        head, tail = out.shape[0], out.shape[1:]
        shape = (head // array.size, array.size) + tail
        return out[:shape[0]*array.size].reshape(shape)

    elif isinstance(array, awkward1._util.listtypes):
        return tonumpy(array.toRegularArray())

    elif isinstance(array, awkward1._util.recordtypes):
        if array.numfields == 0:
            return numpy.empty(len(array), dtype=[])
        contents = [tonumpy(array.field(i)) for i in range(array.numfields)]
        if any(len(x.shape) != 1 for x in contents):
            raise ValueError("cannot convert {0} into numpy.ndarray".format(array))
        out = numpy.empty(len(contents[0]), dtype=[(str(n), x.dtype) for n, x in zip(array.keys(), contents)])
        for n, x in zip(array.keys(), contents):
            out[n] = x
        return out

    elif isinstance(array, awkward1.layout.NumpyArray):
        return numpy.asarray(array)

    elif isinstance(array, awkward1.layout.Content):
        raise AssertionError("unrecognized Content type: {0}".format(type(array)))

    elif isinstance(array, Iterable):
        return numpy.asarray(array)

    else:
        raise ValueError("cannot convert {0} into numpy.ndarray".format(array))

def tolist(array):
    import awkward1.highlevel

    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return array

    elif sys.version_info[0] < 3 and isinstance(array, unicode):
        return array

    elif isinstance(array, numpy.ndarray):
        return array.tolist()

    elif isinstance(array, awkward1.behaviors.string.CharBehavior):
        if array.layout.parameters.get("encoding") is None:
            return array.__bytes__()
        else:
            return array.__str__()

    elif awkward1.operations.describe.parameters(array).get("__array__") == "char":
        if awkward1.operations.describe.parameters(array).get("encoding") is None:
            return awkward1.behaviors.string.CharBehavior(array).__bytes__()
        else:
            return awkward1.behaviors.string.CharBehavior(array).__str__()

    elif isinstance(array, awkward1.highlevel.Array):
        return [tolist(x) for x in array]

    elif isinstance(array, awkward1.highlevel.Record):
        return tolist(array.layout)

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return tolist(array.snapshot())

    elif isinstance(array, awkward1.layout.Record) and array.istuple:
        return tuple(tolist(x) for x in array.fields())

    elif isinstance(array, awkward1.layout.Record):
        return {n: tolist(x) for n, x in array.fielditems()}

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return [tolist(x) for x in array.snapshot()]

    elif isinstance(array, awkward1.layout.NumpyArray):
        return numpy.asarray(array).tolist()

    elif isinstance(array, awkward1.layout.Content):
        return [tolist(x) for x in array]

    elif isinstance(array, dict):
        return dict((n, tolist(x)) for n, x in array.items())

    elif isinstance(array, Iterable):
        return [tolist(x) for x in array]

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

def tojson(array, destination=None, pretty=False, maxdecimals=None, buffersize=65536):
    import awkward1.highlevel

    if array is None or isinstance(array, (bool, str, bytes, numbers.Number)):
        return json.dumps(array)

    elif isinstance(array, bytes):
        return json.dumps(array.decode("utf-8", "surrogateescape"))

    elif sys.version_info[0] < 3 and isinstance(array, unicode):
        return json.dumps(array)

    elif isinstance(array, numpy.ndarray):
        out = awkward1.layout.NumpyArray(array)

    elif isinstance(array, awkward1.highlevel.Array):
        out = array.layout

    elif isinstance(array, awkward1.highlevel.Record):
        out = array.layout

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        out = array.snapshot().layout

    elif isinstance(array, awkward1.layout.Record):
        out = array

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        out = array.snapshot()

    elif isinstance(array, awkward1.layout.Content):
        out = array

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

    if destination is None:
        return out.tojson(pretty=pretty, maxdecimals=maxdecimals)
    else:
        return out.tojson(destination, pretty=pretty, maxdecimals=maxdecimals, buffersize=buffersize)

def tolayout(array, allowrecord=True, allowother=False, numpytype=(numpy.number,)):
    import awkward1.highlevel

    if isinstance(array, awkward1.highlevel.Array):
        return array.layout

    elif allowrecord and isinstance(array, awkward1.highlevel.Record):
        return array.layout

    elif isinstance(array, awkward1.highlevel.ArrayBuilder):
        return array.snapshot().layout

    elif isinstance(array, awkward1.layout.ArrayBuilder):
        return array.snapshot()

    elif isinstance(array, awkward1.layout.Content):
        return array

    elif allowrecord and isinstance(array, awkward1.layout.Record):
        return array

    elif isinstance(array, numpy.ma.MaskedArray):
        mask = numpy.ma.getmaskarray(array).reshape(-1)
        index = numpy.full(len(mask), -1, dtype=numpy.int64)
        index[~mask] = numpy.arange(len(mask) - numpy.count_nonzero(mask), dtype=numpy.int64)
        index = awkward1.layout.Index64(index)
        data = numpy.ma.getdata(array)
        out = awkward1.layout.IndexedOptionArray(index, awkward1.layout.NumpyArray(data.reshape(-1)))
        for size in array.shape[:0:-1]:
            out = awkward1.layout.RegularArray(out, size)
        return out

    elif isinstance(array, numpy.ndarray):
        if not issubclass(array.dtype.type, numpytype):
            raise ValueError("NumPy {0} not allowed".format(repr(array.dtype)))
        out = awkward1.layout.NumpyArray(array.reshape(-1))
        for size in array.shape[:0:-1]:
            out = awkward1.layout.RegularArray(out, size)
        return out

    elif isinstance(array, str) or (awkward1._util.py27 and isinstance(array, unicode)):
        return fromiter([array], highlevel=False)

    elif isinstance(array, Iterable):
        return fromiter(array, highlevel=False)

    elif not allowother:
        raise TypeError("{0} cannot be converted into an Awkward Array".format(array))

    else:
        return array

def regularize_numpyarray(array, allowempty=True, highlevel=True):
    def getfunction(layout):
        if isinstance(layout, awkward1.layout.NumpyArray) and layout.ndim != 1:
            return lambda: layout.toRegularArray()
        elif isinstance(layout, awkward1.layout.EmptyArray) and not allowempty:
            return lambda: layout.toNumpyArray()
        else:
            return None
    out = awkward1._util.recursively_apply(tolayout(array), getfunction)
    if highlevel:
        return awkward1._util.wrap(out, awkward1._util.behaviorof(array))
    else:
        return out

def fromawkward0(array, keeplayout=False, regulararray=False, highlevel=True, behavior=None):
    import awkward as awkward0

    def recurse(array):
        if isinstance(array, dict):
            keys = []
            values = []
            for n, x in array.items():
                keys.append(n)
                if isinstance(x, (dict, tuple, numpy.ma.MaskedArray, numpy.ndarray, awkward0.array.base.AwkwardArray)):
                    values.append(recurse(x)[numpy.newaxis])
                else:
                    values.append(awkward1.layout.NumpyArray(numpy.array([x])))
            return awkward1.layout.RecordArray(values, keys)[0]

        elif isinstance(array, tuple):
            values = []
            for x in array:
                if isinstance(x, (dict, tuple, numpy.ma.MaskedArray, numpy.ndarray, awkward0.array.base.AwkwardArray)):
                    values.append(recurse(x)[numpy.newaxis])
                else:
                    values.append(awkward1.layout.NumpyArray(numpy.array([x])))
            return awkward1.layout.RecordArray(values)[0]

        elif isinstance(array, numpy.ma.MaskedArray):
            return fromnumpy(array, regulararray=regulararray, highlevel=False)

        elif isinstance(array, numpy.ndarray):
            return fromnumpy(array, regulararray=regulararray, highlevel=False)

        elif isinstance(array, awkward0.JaggedArray):
            # starts, stops, content
            # offsetsaliased(starts, stops)
            startsmax = numpy.iinfo(array.starts.dtype.type).max
            stopsmax = numpy.iinfo(array.stops.dtype.type).max
            if len(array.starts.shape) == 1 and len(array.stops.shape) == 1 and awkward0.JaggedArray.offsetsaliased(array.starts, array.stops):
                if startsmax >= fromawkward0.int64max:
                    offsets = awkward1.layout.Index64(array.offsets)
                    return awkward1.layout.ListOffsetArray64(offsets, recurse(array.content))
                elif startsmax >= fromawkward0.uint32max:
                    offsets = awkward1.layout.IndexU32(array.offsets)
                    return awkward1.layout.ListOffsetArrayU32(offsets, recurse(array.content))
                else:
                    offsets = awkward1.layout.Index32(array.offsets)
                    return awkward1.layout.ListOffsetArray32(offsets, recurse(array.content))

            else:
                if startsmax >= fromawkward0.int64max or stopsmax >= fromawkward0.int64max:
                    starts = awkward1.layout.Index64(array.starts.reshape(-1))
                    stops = awkward1.layout.Index64(array.stops.reshape(-1))
                    out = awkward1.layout.ListArray64(starts, stops, recurse(array.content))
                elif startsmax >= fromawkward0.uint32max or stopsmax >= fromawkward0.uint32max:
                    starts = awkward1.layout.IndexU32(array.starts.reshape(-1))
                    stops = awkward1.layout.IndexU32(array.stops.reshape(-1))
                    out = awkward1.layout.ListArrayU32(starts, stops, recurse(array.content))
                else:
                    starts = awkward1.layout.Index32(array.starts.reshape(-1))
                    stops = awkward1.layout.Index32(array.stops.reshape(-1))
                    out = awkward1.layout.ListArray32(starts, stops, recurse(array.content))
                for size in array.starts.shape[:0:-1]:
                    out = awkward1.layout.RegularArray(out, size)
                return out

        elif isinstance(array, awkward0.Table):
            # contents
            if array.istuple:
                return awkward1.layout.RecordArray([recurse(x) for x in array.contents.values()])
            else:
                keys = []
                values = []
                for n, x in array.contents.items():
                    keys.append(n)
                    values.append(recurse(x))
                return awkward1.layout.RecordArray(values, keys)

        elif isinstance(array, awkward0.UnionArray):
            # tags, index, contents
            indexmax = numpy.iinfo(array.index.dtype.type).max
            if indexmax >= fromawkward0.int64max:
                tags = awkward1.layout.Index8(array.tags.reshape(-1))
                index = awkward1.layout.Index64(array.index.reshape(-1))
                out = awkward1.layout.UnionArray8_64(tags, index, [recurse(x) for x in array.contents])
            elif indexmax >= fromawkward0.uint32max:
                tags = awkward1.layout.Index8(array.tags.reshape(-1))
                index = awkward1.layout.IndexU32(array.index.reshape(-1))
                out = awkward1.layout.UnionArray8_U32(tags, index, [recurse(x) for x in array.contents])
            else:
                tags = awkward1.layout.Index8(array.tags.reshape(-1))
                index = awkward1.layout.Index32(array.index.reshape(-1))
                out = awkward1.layout.UnionArray8_32(tags, index, [recurse(x) for x in array.contents])

            for size in array.tags.shape[:0:-1]:
                out = awkward1.layout.RegularArray(out, size)
            return out

        elif isinstance(array, awkward0.MaskedArray):
            # mask, content, maskedwhen
            if keeplayout:
                raise ValueError("awkward1.MaskedArray hasn't been written yet; try keeplayout=False to convert it to the nearest equivalent")
            ismasked = array.boolmask(maskedwhen=True).reshape(-1)
            index = numpy.arange(len(ismasked))
            index[ismasked] = -1
            out = awkward1.layout.IndexedOptionArray64(awkward1.layout.Index64(index), recurse(array.content))

            for size in array.mask.shape[:0:-1]:
                out = awkward1.layout.RegularArray(out, size)
            return out

        elif isinstance(array, awkward0.BitMaskedArray):
            # mask, content, maskedwhen, lsborder
            if keeplayout:
                raise ValueError("awkward1.BitMaskedArray hasn't been written yet; try keeplayout=False to convert it to the nearest equivalent")
            ismasked = array.boolmask(maskedwhen=True)
            index = numpy.arange(len(ismasked))
            index[ismasked] = -1
            return awkward1.layout.IndexedOptionArray64(awkward1.layout.Index64(index), recurse(array.content))

        elif isinstance(array, IndexedMaskedArray):
            # mask, content, maskedwhen
            indexmax = numpy.iinfo(array.index.dtype.type).max
            if indexmax >= fromawkward0.int64max:
                index = awkward1.layout.Index64(array.index.reshape(-1))
                out = awkward1.layout.IndexedOptionArray64(index, recurse(array.content))
            elif indexmax >= fromawkward0.uint32max:
                index = awkward1.layout.IndexU32(array.index.reshape(-1))
                out = awkward1.layout.IndexedOptionArrayU32(index, recurse(array.content))
            else:
                index = awkward1.layout.Index32(array.index.reshape(-1))
                out = awkward1.layout.IndexedOptionArray32(index, recurse(array.content))

            for size in array.tags.shape[:0:-1]:
                out = awkward1.layout.RegularArray(out, size)
            return out

        elif isinstance(array, awkward0.IndexedArray):
            # index, content
            indexmax = numpy.iinfo(array.index.dtype.type).max
            if indexmax >= fromawkward0.int64max:
                index = awkward1.layout.Index64(array.index.reshape(-1))
                out = awkward1.layout.IndexedArray64(index, recurse(array.content))
            elif indexmax >= fromawkward0.uint32max:
                index = awkward1.layout.IndexU32(array.index.reshape(-1))
                out = awkward1.layout.IndexedArrayU32(index, recurse(array.content))
            else:
                index = awkward1.layout.Index32(array.index.reshape(-1))
                out = awkward1.layout.IndexedArray32(index, recurse(array.content))

            for size in array.tags.shape[:0:-1]:
                out = awkward1.layout.RegularArray(out, size)
            return out

        elif isinstance(array, awkward0.SparseArray):
            # length, index, content, default
            if keeplayout:
                raise ValueError("awkward1.SparseArray hasn't been written (if at all); try keeplayout=False to convert it to the nearest equivalent")
            return recurse(array.dense)

        elif isinstance(array, awkward0.StringArray):
            # starts, stops, content, encoding
            raise NotImplementedError

        elif isinstance(array, awkward0.ObjectArray):
            # content, generator, args, kwargs
            raise NotImplementedError

        if isinstance(array, awkward0.ChunkedArray):
            # chunks, chunksizes
            raise NotImplementedError

        elif isinstance(array, awkward0.AppendableArray):
            # chunkshape, dtype, chunks
            raise NotImplementedError

        elif isinstance(array, awkward0.VirtualArray):
            # generator, args, kwargs, cache, persistentkey, type, nbytes, persistvirtual
            raise NotImplementedError

        else:
            raise TypeError("not an awkward0 array: {0}".format(repr(array)))

    out = recurse(array)
    if highlevel:
        return awkward1._util.wrap(out, behavior)
    else:
        return out

fromawkward0.int8max = numpy.iinfo(numpy.int8).max
fromawkward0.int32max = numpy.iinfo(numpy.int32).max
fromawkward0.uint32max = numpy.iinfo(numpy.uint32).max
fromawkward0.int64max = numpy.iinfo(numpy.int64).max

def toawkward0(array, keeplayout=False):
    import awkward as awkward0

    def recurse(layout):
        if isinstance(layout, awkward1.layout.NumpyArray):
            return numpy.asarray(layout)

        elif isinstance(layout, awkward1.layout.EmptyArray):
            return numpy.array([])

        elif isinstance(layout, awkward1.layout.RegularArray):
            # content, size
            if keeplayout:
                raise ValueError("awkward0 has no equivalent of RegularArray; try keeplayout=False to convert it to the nearest equivalent")
            offsets = numpy.arange(0, (len(layout) + 1)*layout.size, layout.size)
            return awkward0.JaggedArray.fromoffsets(offsets, recurse(layout.content))

        elif isinstance(layout, awkward1.layout.ListArray32):
            # starts, stops, content
            return awkward0.JaggedArray(numpy.asarray(layout.starts), numpy.asarray(layout.stops), recurse(layout.content))

        elif isinstance(layout, awkward1.layout.ListArrayU32):
            # starts, stops, content
            return awkward0.JaggedArray(numpy.asarray(layout.starts), numpy.asarray(layout.stops), recurse(layout.content))

        elif isinstance(layout, awkward1.layout.ListArray64):
            # starts, stops, content
            return awkward0.JaggedArray(numpy.asarray(layout.starts), numpy.asarray(layout.stops), recurse(layout.content))

        elif isinstance(layout, awkward1.layout.ListOffsetArray32):
            # offsets, content
            return awkward0.JaggedArray.fromoffsets(numpy.asarray(layout.offsets), recurse(layout.content))

        elif isinstance(layout, awkward1.layout.ListOffsetArrayU32):
            # offsets, content
            return awkward0.JaggedArray.fromoffsets(numpy.asarray(layout.offsets), recurse(layout.content))

        elif isinstance(layout, awkward1.layout.ListOffsetArray64):
            # offsets, content
            return awkward0.JaggedArray.fromoffsets(numpy.asarray(layout.offsets), recurse(layout.content))

        elif isinstance(layout, awkward1.layout.Record):
            # istuple, numfields, field(i)
            out = []
            for i in range(layout.numfields):
                content = layout.field(i)
                if isinstance(content, (awkward1.layout.Content, awkward1.layout.Record)):
                    out.append(recurse(content))
                else:
                    out.append(content)
            if layout.istuple:
                return tuple(out)
            else:
                return dict(zip(layout.keys(), out))

        elif isinstance(layout, awkward1.layout.RecordArray):
            # istuple, numfields, field(i)
            if layout.numfields == 0 and len(layout) != 0:
                raise ValueError("cannot convert zero-field, nonzero-length RecordArray to awkward0.Table (limitation in awkward0)")
            keys = layout.keys()
            values = [recurse(x) for x in layout.contents]
            pairs = collections.OrderedDict(zip(keys, values))
            out = awkward0.Table(pairs)
            if layout.istuple:
                out._rowname = "tuple"
            return out

        elif isinstance(layout, awkward1.layout.UnionArray8_32):
            # tags, index, numcontents, content(i)
            return awkward0.UnionArray(numpy.asarray(layout.tags), numpy.asarray(layout.index), [recurse(x) for x in layout.contents])

        elif isinstance(layout, awkward1.layout.UnionArray8_U32):
            # tags, index, numcontents, content(i)
            return awkward0.UnionArray(numpy.asarray(layout.tags), numpy.asarray(layout.index), [recurse(x) for x in layout.contents])

        elif isinstance(layout, awkward1.layout.UnionArray8_64):
            # tags, index, numcontents, content(i)
            return awkward0.UnionArray(numpy.asarray(layout.tags), numpy.asarray(layout.index), [recurse(x) for x in layout.contents])

        elif isinstance(layout, awkward1.layout.IndexedOptionArray32):
            # index, content
            index = numpy.asarray(layout.index)
            toosmall = (index < -1)
            if toosmall.any():
                index = index.copy()
                index[toosmall] = -1
            return awkward0.IndexedMaskedArray(index, recurse(layout.content))

        elif isinstance(layout, awkward1.layout.IndexedOptionArray64):
            # index, content
            index = numpy.asarray(layout.index)
            toosmall = (index < -1)
            if toosmall.any():
                index = index.copy()
                index[toosmall] = -1
            return awkward0.IndexedMaskedArray(index, recurse(layout.content))

        elif isinstance(layout, awkward1.layout.IndexedArray32):
            # index, content
            return awkward0.IndexedArray(numpy.asarray(layout.index), recurse(layout.content))

        elif isinstance(layout, awkward1.layout.IndexedArrayU32):
            # index, content
            return awkward0.IndexedArray(numpy.asarray(layout.index), recurse(layout.content))

        elif isinstance(layout, awkward1.layout.IndexedArray64):
            # index, content
            return awkward0.IndexedArray(numpy.asarray(layout.index), recurse(layout.content))

        else:
            raise AssertionError("missing converter for {0}".format(type(layout).__name__))

    layout = tolayout(array, allowrecord=True, allowother=False, numpytype=(numpy.generic,))
    return recurse(layout)

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numbers", "json", "Iterable", "numpy", "awkward1")]

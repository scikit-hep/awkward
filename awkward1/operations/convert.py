# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import sys
import numbers
import json
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy

import awkward1._util
import awkward1.layout

def fromnumpy(array):
    def recurse(array):
        if len(array.shape) == 0:
            return awkward1.layout.NumpyArray(array.reshape(1))
        elif len(array.shape) == 1:
            return awkward1.layout.NumpyArray(array)
        else:
            return awkward1.layout.RegularArray(recurse(array.reshape((-1,) + array.shape[2:])), array.shape[1])
    return awkward1._util.wrap(recurse(array), awkward1.classes, awkward1.functions)

def fromiter(iterable, initial=1024, resize=2.0):
    out = awkward1.layout.FillableArray(initial=initial, resize=resize)
    for x in iterable:
        out.fill(x)
    return awkward1._util.wrap(out.snapshot(), awkward1.classes, awkward1.functions)

def fromjson(source, initial=1024, resize=2.0, buffersize=65536):
    return awkward1._util.wrap(awkward1.layout.fromjson(source, initial=initial, resize=resize, buffersize=buffersize), awkward1.classes, awkward1.functions)

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

    elif isinstance(array, awkward1.highlevel.FillableArray):
        return tonumpy(array.snapshot().layout)

    elif isinstance(array, awkward1.layout.FillableArray):
        return tonumpy(array.snapshot())

    elif awkward1.operations.describe.parameters(array).get("__class__") == "char":
        if awkward1.operations.describe.parameters(array).get("encoding") is None:
            return tonumpy(array.__bytes__())
        else:
            return tonumpy(array.__str__())

    elif awkward1.operations.describe.parameters(array).get("__class__") == "string":
        if awkward1.operations.describe.parameters(array.content).get("encoding") is None:
            return numpy.array([awkward1.behavior.string.CharBehavior(array[i]).__bytes__() for i in range(len(array))])
        else:
            return numpy.array([awkward1.behavior.string.CharBehavior(array[i]).__str__() for i in range(len(array))])

    elif isinstance(array, awkward1.layout.EmptyArray):
        return numpy.array([])

    elif isinstance(array, (awkward1.layout.UnionArray8_32, awkward1.layout.UnionArray8_U32, awkward1.layout.UnionArray8_64)):
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

    elif isinstance(array, (awkward1.layout.IndexedOptionArray32, awkward1.layout.IndexedOptionArray64)):
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

    elif isinstance(array, (awkward1.layout.ListArray32, awkward1.layout.ListArrayU32, awkward1.layout.ListArray64, awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArrayU32, awkward1.layout.ListOffsetArray64)):
        return tonumpy(array.toRegularArray())

    elif isinstance(array, awkward1.layout.RecordArray):
        if array.numfields == 0:
            return numpy.empty(len(array), dtype=[])
        contents = [tonumpy(array.field(i)) for i in range(array.numfields)]
        if any(len(x.shape) != 1 for x in contents):
            raise ValueError("cannot convert {0} into numpy.ndarray".format(array))
        out = numpy.empty(len(contents[0]), dtype=[(n, x.dtype) for n, x in zip(array.keys(), contents)])
        for n, x in zip(array.keys(), contents):
            out[n] = x
        return out

    elif isinstance(array, awkward1.layout.NumpyArray):
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

    elif isinstance(array, awkward1.behavior.string.CharBehavior):
        if array.layout.parameters.get("encoding") is None:
            return array.__bytes__()
        else:
            return array.__str__()

    elif awkward1.operations.describe.parameters(array).get("__class__") == "char":
        if awkward1.operations.describe.parameters(array).get("encoding") is None:
            return awkward1.behavior.string.CharBehavior(array).__bytes__()
        else:
            return awkward1.behavior.string.CharBehavior(array).__str__()

    elif isinstance(array, awkward1.highlevel.Array):
        return [tolist(x) for x in array]

    elif isinstance(array, awkward1.highlevel.Record):
        return tolist(array.layout)

    elif isinstance(array, awkward1.highlevel.FillableArray):
        return tolist(array.snapshot())

    elif isinstance(array, awkward1.layout.Record) and array.istuple:
        return tuple(tolist(x) for x in array.fields())

    elif isinstance(array, awkward1.layout.Record):
        return {n: tolist(x) for n, x in array.fielditems()}

    elif isinstance(array, awkward1.layout.FillableArray):
        return [tolist(x) for x in array.snapshot()]

    elif isinstance(array, awkward1.layout.NumpyArray):
        return numpy.asarray(array).tolist()

    elif isinstance(array, awkward1.layout.Content):
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

    elif isinstance(array, awkward1.highlevel.FillableArray):
        out = array.snapshot().layout

    elif isinstance(array, awkward1.layout.Record):
        out = array

    elif isinstance(array, awkward1.layout.FillableArray):
        out = array.snapshot()

    elif isinstance(array, awkward1.layout.Content):
        out = array

    else:
        raise TypeError("unrecognized array type: {0}".format(repr(array)))

    if destination is None:
        return out.tojson(pretty=pretty, maxdecimals=maxdecimals)
    else:
        return out.tojson(destination, pretty=pretty, maxdecimals=maxdecimals, buffersize=buffersize)

__all__ = [x for x in list(globals()) if not x.startswith("_") and x not in ("numbers", "json", "Iterable", "numpy", "awkward1")]
